import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse


def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rev: torch.Tensor,
    k_rev: torch.Tensor,
    v_rev: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)
    comm_rev = RingComm(process_group)
    rank = comm.rank
    world_size = comm.world_size
    rank_rev = comm.world_size - rank - 1

    out = None
    lse = None
    block_out = None
    block_lse = None
    next_block_out = None
    next_block_lse = None
    next_q = None

    out_rev = None
    lse_rev = None
    block_out_rev = None
    block_lse_rev = None
    next_block_out_rev = None
    next_block_lse_rev = None
    next_q_rev = None

    for step in range(world_size):

        # origin send q
        if rank > 0 and rank <= world_size - 1 - step:
            comm.send_backward(q, 1)
        
        if rank >= 0 and rank < world_size - 1 - step:
            next_q: torch.Tensor = comm.recv_forward(q, 1)

        if comm._ops_backward:
            comm.commit_backward()

        # reverse send q
        if rank_rev > 0 and rank_rev <= world_size - 1 - step:
            comm_rev.send_forward(q_rev, 1)
        
        if rank_rev >= 0 and rank_rev < world_size - 1 - step:
            next_q_rev: torch.Tensor = comm_rev.recv_backward(q_rev, 1)

        if comm_rev._ops_forward:
            comm_rev.commit_forward()

        # origin send out and lse 
        if step > 1:
            if rank <= world_size - step:
                comm.send_forward(block_out, step - 1)
                comm.send_forward(block_lse, step - 1)
            if rank >= step - 1 and rank < world_size:
                next_block_out: torch.Tensor = comm.recv_backward(block_out, step - 1)
                next_block_lse: torch.Tensor = comm.recv_backward(block_lse, step - 1)
        
        # reverse send out and lse 
        if step > 1:
            if rank_rev <= world_size - step:
                comm_rev.send_backward(block_out_rev, step - 1)
                comm_rev.send_backward(block_lse_rev, step - 1)
            if rank_rev >= step - 1 and rank_rev < world_size:
                next_block_out_rev: torch.Tensor = comm_rev.recv_forward(block_out_rev, step - 1)
                next_block_lse_rev: torch.Tensor = comm_rev.recv_forward(block_lse_rev, step - 1)
        
        # origin commit
        if comm._ops_forward:
            comm.commit_forward()

        # reverse commit
        if comm_rev._ops_backward:
            comm_rev.commit_backward()

        # origin compute
        if not causal or step <= (world_size - rank - 1):

            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
        
        # reverse compute
        if not causal or step <= (world_size - rank_rev - 1):

            block_out_rev, _, _, _, _, block_lse_rev, _, _ = _flash_attn_forward(
                q_rev,
                k_rev,
                v_rev,
                dropout_p,
                softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )


            # print(f"Step {step} rank {rank} Compute succeed ___________________________________________")
        
        # origin wait
        if comm._ops_backward:
            comm.wait_backward()
            q = next_q

        if comm._ops_forward:
            comm.wait_forward()

        # rev wait
        if comm_rev._ops_forward:
            comm_rev.wait_forward()
            q_rev = next_q_rev

        if comm_rev._ops_backward:
            comm_rev.wait_backward()

        # origin update
        if step == 0:
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
        if step > 1 and step <= rank + 1:
            out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)

        # reverse update
        if step == 0:
            out_rev, lse_rev = update_out_and_lse(out_rev, lse_rev, block_out_rev, block_lse_rev)
        
        if step > 1 and step <= rank_rev + 1:
            out_rev, lse_rev = update_out_and_lse(out_rev, lse_rev, next_block_out_rev, next_block_lse_rev)

    # final origin send
    if rank == 0:
        comm.send_forward(block_out, world_size - 1) 
        comm.send_forward(block_lse, world_size - 1) 
    
    # final reverse send 
    if rank_rev == 0:
        comm_rev.send_backward(block_out_rev, world_size - 1) 
        comm_rev.send_backward(block_lse_rev, world_size - 1) 

    # final origin recv
    if rank == world_size - 1:
        next_block_out: torch.Tensor = comm.recv_backward(block_out, world_size - 1)    
        next_block_lse: torch.Tensor = comm.recv_backward(block_lse, world_size - 1)
    
    # final reverse recv
    if rank_rev == world_size - 1:
        next_block_out_rev: torch.Tensor = comm_rev.recv_forward(block_out_rev, world_size - 1)    
        next_block_lse_rev: torch.Tensor = comm_rev.recv_forward(block_lse_rev, world_size - 1)

    # final origin commit
    if rank == 0 or rank == world_size - 1:
        comm.commit_forward()
    # final reverse commit 
    if rank_rev == 0 or rank_rev == world_size - 1:
        comm_rev.commit_backward()

    # final origin wait
    if rank == 0 or rank == world_size - 1:
        comm.wait_forward()
    # final rev wait
    if rank_rev == 0 or rank_rev == world_size - 1:
        comm_rev.wait_backward()

    # final origin update
    if rank == world_size - 1:
        out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)
    # final reverse update
    if rank_rev == world_size - 1:
        out_rev, lse_rev = update_out_and_lse(out_rev, lse_rev, next_block_out_rev, next_block_lse_rev)

    out = out.to(v.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)

    out_rev = out_rev.to(v_rev.dtype)
    lse_rev = lse_rev.squeeze(dim=-1).transpose(1, 2)

    return out, lse, out_rev, lse_rev


def ring_flash_attn_forward_inverse(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group) 
    
    world_size = comm.world_size
    # rank = world_size - 1 - comm.rank
    rank = comm.rank
    # print(f"rank {rank} init ___________________________________________")
    out = None
    lse = None

    block_out = None
    block_lse = None

    next_block_out = None
    next_block_lse = None

    block_out = None
    block_lse = None

    next_q = None
    # print(f"process group is {process_group} ")
    # counter = 0
    for step in range(world_size):

        if rank > 0 and rank <= world_size - 1 - step:
            comm.send_backward(q)
            # print(f"rank {rank} send succeed")
        
        if rank >= 0 and rank < world_size - 1 - step:
            next_q: torch.Tensor = comm.recv_forward(q)
            # print(f"rank {rank} recv succeed")
        if comm._ops_backward:
            comm.commit_backward()
            # print("commit")

        # if rank >= 0 and rank < world_size - 1 - step:
        #     count += 1
        # if comm._ops_backward:
        #     count += 1

        if step > 1:
            if rank <= world_size - step:
                comm.send_forward(block_out, step - 1)
                comm.send_forward(block_lse, step - 1)
            if rank >= step - 1 and rank < world_size:
                next_block_out: torch.Tensor = comm.recv_backward(block_out, step - 1)
                next_block_lse: torch.Tensor = comm.recv_backward(block_lse, step - 1)
        
        
        if comm._ops_forward:
            comm.commit_forward()

        if not causal or step <= (world_size - rank - 1):

            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )


            # print(f"Step {step} rank {rank} Compute succeed ___________________________________________")

        if comm._ops_backward:
            comm.wait_backward()
            q = next_q

        # if not causal or step <= (world_size - rank - 1):
            # count += 1
        
        if comm._ops_forward:
            comm.wait_forward()
        
        # if not causal or step <= (world_size - rank - 1):
            # count += 1

        if step == 0:
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
        if step > 1 and step <= rank + 1:
            out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)

        # if comm._ops_forward:
        #     count += 1

        # if step > 1 and step <= rank + 1:
        #     count += 1

    if rank == 0:
        comm.send_forward(block_out, world_size - 1) 
        comm.send_forward(block_lse, world_size - 1) 

    if rank == world_size - 1:
        next_block_out: torch.Tensor = comm.recv_backward(block_out, world_size - 1)    
        next_block_lse: torch.Tensor = comm.recv_backward(block_lse, world_size - 1)
    
    if rank == 0 or rank == world_size - 1:
        comm.commit_forward()
        comm.wait_forward()

    if rank == world_size - 1:
        out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)


    out = out.to(v.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        q_rev,
        k_rev,
        v_rev,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()

        k_rev = k_rev.contiguous()
        v_rev = v_rev.contiguous()


        out, softmax_lse, out_rev, lse_rev = ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            q_rev,
            k_rev,
            v_rev,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # out, softmax_lse = ring_flash_attn_forward_inverse(
        #     group,
        #     q,
        #     k,
        #     v,
        #     softmax_scale=softmax_scale,
        #     dropout_p=dropout_p,
        #     causal=causal,
        #     window_size=window_size,
        #     alibi_slopes=alibi_slopes,
        #     deterministic=False,
        # )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        # return out, out_rev if not return_softmax else (out, softmax_lse, out_rev, lse_rev)
        return out, softmax_lse, out_rev, lse_rev


def ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def ring_flash_attn_func(
    q,
    k,
    v,
    q_rev,
    k_rev,
    v_rev,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        q_rev,
        k_rev,
        v_rev,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )

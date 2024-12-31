import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse
import copy
import nvtx
import os


def zigzag_ring_flash_attn_forward(
    process_q_group,
    process_out_group,  
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
    assert causal == True, "zigzag ring is meaningless for causal=False"
    q_comm = RingComm(process_q_group)
    out_comm = RingComm(process_out_group)

    rank = q_comm.rank
    world_size = q_comm.world_size

    block_seq_len = q.shape[1] // 2

    k0 = k[:, :block_seq_len]
    v0 = v[:, :block_seq_len]

    out = None
    # (seqlen, nheads, headdim)

    lse = None
    # (nheads, seq_len)

    block_out = None
    block_lse = None

    block_out_all = None
    block_out_half = None
    block_lse_all = None
    block_lse_half = None

    next_q = None
    next_block_out = None
    next_block_lse = None

    def forward(q, k, v, causal):
        # _ = torch.zeros(1, device = k.device)
        # _ = torch.zeros(1, device = k.device)
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=0,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    for step in range(world_size):
        # Send Q，0->3->2->1
        with nvtx.annotate("Zigzag Q comm"):
            if step < world_size - 1:
                next_rank = (rank - 1 + world_size) % world_size

                if next_rank + 1 + step < world_size:
                    q_comm.send_backward(q, 1)
                
                else:
                    q_comm.send_backward(q[:, -block_seq_len:], 1)

                if rank + step < world_size - 1:  
                    next_q: torch.Tensor = q_comm.recv_forward(q, 1)

                else:
                    next_q: torch.Tensor = q_comm.recv_forward(q[:, -block_seq_len:], 1)

                q_comm.commit_backward()

        with nvtx.annotate("out comm"):
            if step > 1:
                out_comm.send_forward(block_out, step - 1)
                out_comm.send_forward(block_lse, step - 1)

                if rank < step - 1:
                    next_block_out: torch.Tensor = out_comm.recv_backward(block_out_half, step - 1)
                    next_block_lse: torch.Tensor = out_comm.recv_backward(block_lse_half, step - 1)            
          
                else:
                    next_block_out: torch.Tensor = out_comm.recv_backward(block_out_all, step - 1)
                    next_block_lse: torch.Tensor = out_comm.recv_backward(block_lse_all, step - 1)   

                out_comm.commit_forward()

        with nvtx.annotate("compute"):
            if step == 0:
                block_out, block_lse = forward(q, k, v, causal=True)

                block_out_all = block_out
                block_out_half = block_out[:, block_seq_len:]

                block_lse_all = block_lse
                block_lse_half = block_lse[:, :, block_seq_len:]

                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            elif rank < world_size - step:

                block_out, block_lse = forward(q, k0, v0, causal=False)

            else:
                block_out, block_lse = forward(q, k, v, causal=False)


        # Q Wait
        if step + 1 != world_size:
            q_comm.wait_backward()
            q = next_q

        if step > 1:
            out_comm.wait_forward()

        with nvtx.annotate("update"):
            if step > 1:
                if rank < step - 1:
                    out, lse = update_out_and_lse(
                        out,
                        lse,
                        next_block_out,
                        next_block_lse,
                        slice_=(slice(None), slice(block_seq_len, None)),
                    )

                else:
                    out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)      
 
    
    
    with nvtx.annotate("last comm"):
        out_comm.send_forward(block_out, world_size - 1)
        out_comm.send_forward(block_lse, world_size - 1)

        if rank != world_size - 1:
            next_block_out: torch.Tensor = out_comm.recv_backward(block_out_half, world_size - 1)
            next_block_lse: torch.Tensor = out_comm.recv_backward(block_lse_half, world_size - 1)
        else:
            next_block_out: torch.Tensor = out_comm.recv_backward(block_out_all, world_size - 1)
            next_block_lse: torch.Tensor = out_comm.recv_backward(block_lse_all, world_size - 1)

        out_comm.commit_forward()

        out_comm.wait_forward()
    with nvtx.annotate("last update"):
        if rank != world_size - 1:
            out, lse = update_out_and_lse(
                        out,
                        lse,
                        next_block_out,
                        next_block_lse,
                        slice_=(slice(None), slice(block_seq_len, None)),
                    ) 
        else:
            out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        q_group,
        out_group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = zigzag_ring_flash_attn_forward(
            q_group,
            out_group,
            q,
            k,
            v,
            softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.q_group = q_group
        return out if not return_softmax else (out, softmax_lse, None)


def zigzag_ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    q_group=None,
    out_group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
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
        q_group,
        out_group,
    )


def zigzag_ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    q_group=None,
    out_group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
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
        q_group,
        out_group,
    )


def zigzag_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    q_group=None,
    out_group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        q_group,
        out_group,
    )

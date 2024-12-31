import torch
from flash_attn.flash_attn_interface import _flash_attn_forward
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from yunchang.ring.utils import RingComm, update_out_and_lse
from yunchang.ring.ring_flash_attn import RingFlashAttnFunc


def ring_flash_attn_forward(
    process_q_group,
    process_out_group, 
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
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
):
    supported_joint_strategy = ["none", "front", "rear"]
    if joint_strategy not in supported_joint_strategy:
        raise ValueError(
            f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
        )
    elif joint_strategy != "none" and (
        joint_tensor_key is None or joint_tensor_value is None
    ):
        raise ValueError(
            f"joint_tensor_key & joint_tensor_value must not be None when joint_strategy is not None"
        )

    comm = RingComm(process_group)
    q_comm = RingComm(process_q_group)
    out_comm = RingComm(process_out_group)

    out = None
    lse = None

    next_k, next_v = None, None

    next_q = None
    next_block_out = None
    next_block_lse = None

    if attn_layer is not None:
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()
        q = q.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:

            q_comm.send_forward(q, 1)

            next_q: torch.Tensor = q_comm.recv_backward(q, 1)

            q_comm.commit_forward()

        if step > 1:
            out_comm.send_backward(block_out, step - 1)
            out_comm.send_backward(block_lse, step - 1)

            next_block_out: torch.Tensor = out_comm.recv_forward(block_out, step - 1)
            next_block_lse: torch.Tensor = out_comm.recv_forward(block_lse, step - 1)
            out_comm.commit_backward()

        _ = torch.zeros(1, device = k.device)
        _ = torch.zeros(1, device = k.device)
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal and step == 0,
            window_size=window_size,
            softcap=0.0,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )

        if step + 1 != comm.world_size:

            q_comm.wait_forward()

            q = next_q

        if step > 1:
            out_comm.wait_backward()
            

        if step == 0:
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step > 1:
            out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)

    out_comm.send_backward(block_out, comm.world_size - 1)
    out_comm.send_backward(block_lse, comm.world_size - 1)

    next_block_out: torch.Tensor = out_comm.recv_forward(block_out, comm.world_size - 1)
    next_block_lse: torch.Tensor = out_comm.recv_forward(block_lse, comm.world_size - 1)

    out_comm.commit_backward()
    out_comm.wait_backward()

    out, lse = update_out_and_lse(out, lse, next_block_out, next_block_lse)

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class xFuserRingFlashAttnFunc(RingFlashAttnFunc):
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
        group,
        q_group,
        out_group,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        out, softmax_lse = ring_flash_attn_forward(
            group,
            q_group,
            out_group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)


def ring_flash_attn_func(
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
    group=None,
    q_group=None,
    out_group=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
):
    return xFuserRingFlashAttnFunc.apply(
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
        group,
        q_group,
        out_group,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )

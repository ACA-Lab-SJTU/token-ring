import os
import sys
directory_to_add = '/root/lms/token_ring'
sys.path.append(directory_to_add)
# os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'


from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch
import torch.distributed as dist
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from token_ring import zigzag_ring_flash_attn_qkvpacked_func, zigzag_ring_flash_attn_func
import nvtx
from torch.distributed import ProcessGroupNCCL

import random


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)             
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.contiguous()

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


if __name__ == "__main__":

    pg_options = ProcessGroupNCCL.Options()
    pg_options.is_high_priority_stream = True 

    dist.init_process_group(backend="nccl", pg_options=pg_options)
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.float16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 24000
    nheads = 32
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False
    n_rep = 4

    assert causal
    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = extract_local(q, rank, world_size).detach().clone()
    local_q.requires_grad = True

    local_k = extract_local(k, rank, world_size).detach().clone()
    local_k.requires_grad = True

    local_v = extract_local(v, rank, world_size).detach().clone()
    local_v.requires_grad = True

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = extract_local(out, rank, world_size)
    local_lse = extract_local(lse, rank, world_size, dim=2)

    q_group = dist.new_group(backend="nccl", pg_options=pg_options)
    out_group = dist.new_group(backend="nccl", pg_options=pg_options)


    with nvtx.annotate("warm up"):
        for _ in range(5):
            ring_out, ring_lse, _ = zigzag_ring_flash_attn_func(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=True,
                q_group=q_group,
                out_group=out_group,
            )
    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# start:")
        print("#" * 30)
        
    with nvtx.annotate(message="Zigzag Ring Flash Attention", domain="test"):
        test = nvtx.start_range(message="test",color="blue")
        ring_out, ring_lse, _ = zigzag_ring_flash_attn_func(
                        local_q,
                        local_k,
                        local_v,
                        dropout_p=dropout_p,
                        causal=causal,
                        window_size=(-1, -1),
                        alibi_slopes=None,
                        deterministic=deterministic,
                        return_attn_probs=True,
                        q_group=q_group,
                        out_group=out_group,
                    )
        nvtx.end_range(test)

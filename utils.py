from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = ["update_out_and_lse", "RingComm"]

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops_forward = []
        self._ops_backward = []
        self.rank = dist.get_rank(self._process_group)

        self.world_size = dist.get_world_size(self._process_group)
        self._reqs_forward = None
        self._reqs_backward = None

        self.send_rank_forward = (self.rank + 1) % self.world_size
        self.send_rank_backward = (self.rank - 1) % self.world_size
        self.recv_rank_forward = (self.rank + 1) % self.world_size
        self.recv_rank_backward = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank_forward = dist.get_global_rank(self._process_group, self.send_rank_forward)
            self.send_rank_backward = dist.get_global_rank(self._process_group, self.send_rank_backward)
            self.recv_rank_forward = dist.get_global_rank(self._process_group, self.recv_rank_forward)
            self.recv_rank_backward = dist.get_global_rank(self._process_group, self.recv_rank_backward)

    def send_forward(self, to_send: torch.Tensor, step: int):
        send_rank_forward = (self.rank + step) % self.world_size
        if self._process_group is not None:
            send_rank_forward = dist.get_global_rank(self._process_group, send_rank_forward)
        send_forward_op = dist.P2POp(
            dist.isend, to_send, send_rank_forward, group=self._process_group
        )
        self._ops_forward.append(send_forward_op)

    def recv_backward(self, to_send: torch.Tensor, step: int, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        recv_rank_backward = (self.rank - step) % self.world_size
        if self._process_group is not None:
            recv_rank_backward = dist.get_global_rank(self._process_group, recv_rank_backward)

        recv_backward_op = dist.P2POp(
            dist.irecv, res, recv_rank_backward, group=self._process_group
        )
        self._ops_forward.append(recv_backward_op)
        return res
    
    def send_backward(self, to_send: torch.Tensor, step: int,):
        send_rank_backward = (self.rank - step) % self.world_size

        if self._process_group is not None:
            send_rank_backward = dist.get_global_rank(self._process_group, send_rank_backward)

        send_backward_op = dist.P2POp(
            dist.isend, to_send, send_rank_backward, group=self._process_group
        )
        self._ops_backward.append(send_backward_op)
    
    def recv_forward(self, to_send: torch.Tensor, step: int, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        recv_rank_forward = (self.rank + step) % self.world_size

        if self._process_group is not None:
            recv_rank_forward = dist.get_global_rank(self._process_group, recv_rank_forward)

        recv_forward_op = dist.P2POp(
            dist.irecv, res, recv_rank_forward, group=self._process_group
        )
        self._ops_backward.append(recv_forward_op)
        return res

    def commit_forward(self):
        if self._reqs_forward is not None:
            raise RuntimeError("commit called twice")
        self._reqs_forward = dist.batch_isend_irecv(self._ops_forward)

    def commit_backward(self):
        if self._reqs_backward is not None:
            raise RuntimeError("commit called twice")
        self._reqs_backward = dist.batch_isend_irecv(self._ops_backward)

    def wait_forward(self):
        if self._reqs_forward is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs_forward:
            req.wait()
        self._reqs_forward = None
        self._ops_forward = []

    def wait_backward(self):
        if self._reqs_backward is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs_backward:
            req.wait()
        self._reqs_backward = None
        self._ops_backward = []


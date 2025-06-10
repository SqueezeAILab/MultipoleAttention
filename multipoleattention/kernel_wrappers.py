import math
import numpy as np
import pytest
import torch
import triton
import triton.language as tl
import time
from multipoleattention.kernels import _fwd_kernel_qk_gen, _reduce_kernel_qk_gen, _fwd_centroid_simple_kernel_qk, _fwd_centroid_kernel_attn, _reduce_kernel_attn, compact_indices_kernel

class _attention_qk_gen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, k_idx, Out, Mout, Lout, head_start_block, head_kv_len, num_kv_blocks, sm_scale):

        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        gpu_name = torch.cuda.get_device_name(q.device)  # Assuming you're checking the first GPU
        if "A6000" in gpu_name or "A5000" in gpu_name:
            BLOCK_N = 64
            BLOCK_M = 64
        elif "A100" in gpu_name or "H100" in gpu_name or "L40S" in gpu_name:
            BLOCK_N = 128
            BLOCK_M = 128
        else:
            print(f"GPU not supported: {gpu_name}")
            assert(False)
        BLOCK_KV = 2048

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {64, 128}

        # num_kv_heads
        NUM_HEADS = q.shape[1] * q.shape[2]
        NUM_KV_HEADS = k.shape[1]
        GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS

        assert(q.shape[1] == NUM_KV_HEADS)
        assert(q.shape[2] == GQA_FACTOR)
        assert(q.shape[0] == 1)
        grid = (1, num_kv_blocks)
        num_warps = 4 if Lk <= 64 else 8

        # create o_tmp to hold temporary outputs
        o_tmp = torch.empty((q.shape[0], num_kv_blocks, GQA_FACTOR, q.shape[3]), device=q.device, dtype=torch.bfloat16)

        # L - running sum, M - running max
        L = torch.empty((num_kv_blocks, GQA_FACTOR), device=q.device, dtype=torch.float32)
        M = torch.empty((num_kv_blocks, GQA_FACTOR), device=q.device, dtype=torch.float32)

        _fwd_kernel_qk_gen[grid](
            q, k, v, k_idx,
            sm_scale, o_tmp,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            k_idx.stride(0), k_idx.stride(1), k_idx.stride(2),
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3),
            head_start_block, head_kv_len,
            L, M,
            q.shape[0], q.shape[1], q.shape[2], k.shape[2], k_idx.shape[2], NUM_KV_HEADS,
            NUMHEADS=q.shape[1],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        # Reduction Kernel
        grid2 = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        _reduce_kernel_qk_gen[grid2](
            o_tmp, Out,
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            head_start_block, head_kv_len,
            L, M,
            Lout, Mout,
            q.shape[2], NUM_KV_HEADS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        return Out, Mout, Lout

class _centroid_lookup_gen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, centroid_labels, num_keys, sm_scale, budget, num_key_value_groups):

        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        gpu_name = torch.cuda.get_device_name(q.device)  # Assuming you're checking the first GPU
        if "A6000" in gpu_name or "A5000" in gpu_name:
            BLOCK_N = 64
            BLOCK_M = 64
        elif "A100" in gpu_name or "H100" in gpu_name or "L40S" in gpu_name:
            BLOCK_N = 128
            BLOCK_M = 128
        else:
            print(f"GPU not supported: {gpu_name}")
            assert(False)
        BLOCK_KV = 2048

        # number of KV heads and GQA factor
        NUM_HEADS = q.shape[1] * q.shape[2]
        NUM_KV_HEADS = k.shape[1]
        GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS

        assert(q.shape[1] == NUM_KV_HEADS)
        assert(q.shape[2] == GQA_FACTOR)

        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]
        assert Lq == Lk
        assert Lk in {64, 128}

        # split along kv dimension
        num_kv_blocks = triton.cdiv(k.shape[2], BLOCK_KV)
        num_query_blocks = triton.cdiv(q.shape[2], BLOCK_M)

        # set up grid - need shape to be ( ceil(q_seqlen / BLOCKSIZE) , NUMHEADS * ceil(nonzero_kv_seqlen_per_head / BLOCKSIZE) )
        assert(q.shape[0] == 1) # not supporting batched inference yet
        grid = (1, q.shape[1], num_kv_blocks)
        num_warps = 4 if Lk <= 64 else 8

        # L - running sum, M - running max
        o = torch.empty((q.shape[0], q.shape[1], k.shape[2], q.shape[2]), device=q.device, dtype=torch.float32)
        score_avg = torch.empty((q.shape[0], q.shape[1], k.shape[2], num_query_blocks), device=q.device, dtype=torch.float32)

        # save query-key dot products for second stage
        qk = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=torch.bfloat16)


        # compute kernel
        _fwd_centroid_simple_kernel_qk[grid](
            q, k, qk,
            num_keys, centroid_labels,
            sm_scale,
            o, score_avg,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            score_avg.stride(0), score_avg.stride(1), score_avg.stride(2), score_avg.stride(3),
            qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
            q.shape[0], q.shape[1], q.shape[2], centroid_labels.shape[2], k.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        # reshape to aggregate across kv heads
        avg_score = score_avg.mean(dim=-1)

        # gather scores
        avg_score_per_token = torch.gather(avg_score, 2, centroid_labels)

        # topk
        num_tokens_retained = budget

        # sort scores
        avg_score_sorted, avg_score_sorted_idx = avg_score.sort(dim=2, descending=True)

        # sort num_keys
        num_keys_sorted = num_keys.gather(
            dim=2, index=avg_score_sorted_idx
        )
        num_keys_sorted_cumsum = num_keys_sorted.cumsum(dim=2)

        # compute sorted mask
        exceed_mask = num_keys_sorted_cumsum > num_tokens_retained
        first_exceed_idx = exceed_mask.float().argmax(dim=2)

        # extra guard - exceed_mask can be 0 if threshold=1
        all_zero_mask = ~exceed_mask.any(dim=2)
        first_exceed_idx[all_zero_mask] = exceed_mask.shape[2] - 1
        first_exceed_idx = first_exceed_idx.unsqueeze(-1)

        # get and apply threshold
        score_threshold = avg_score_sorted.gather(dim=2, index=first_exceed_idx)

        # strict threshold (to line up with kernel benchmarking)
        mask = (avg_score_per_token > score_threshold)
        centroid_mask = (avg_score > score_threshold)

        return mask, centroid_mask, qk

def compute_k_idx_optimized(keep_tensor):
    BLOCK_KV = 2048
    BLOCK_SIZE = 4096
    B, H, N_CTX_KV = keep_tensor.shape

    # calculate number of blocks needed
    num_blocks = (N_CTX_KV + BLOCK_SIZE - 1) // BLOCK_SIZE

    # set up grid
    grid = (B, H, num_blocks)

    # prepare output tensors
    indices = torch.empty_like(keep_tensor, dtype=torch.int64)
    num_kv_per_head = torch.zeros((B, H), dtype=torch.int32, device=keep_tensor.device)

    # launch kernel for computing indices
    compact_indices_kernel[grid](
        keep_tensor,
        indices,
        num_kv_per_head,
        N_CTX_KV,
        H,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

    # extra metadata needed for sparse flash attn
    num_kv_blocks_per_head = num_kv_per_head.float().div(BLOCK_KV).ceil().int()
    num_kv_blocks_per_head_csum = num_kv_blocks_per_head.cumsum(dim=-1).squeeze(0)
    num_kv_blocks = num_kv_blocks_per_head_csum[-1].clone()
    head_start_block = num_kv_blocks_per_head_csum - num_kv_blocks_per_head
    return indices, num_kv_per_head, num_kv_blocks, head_start_block

class _centroid_replacement(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, v, centroid_mask, num_keys, sm_scale, qk, o, m, l):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK_N = 64
        BLOCK_M = 64
        BLOCK_KV = 512

        # shape constraints
        Lq = q.shape[-1]
        assert Lq in {64, 128}

        # split along kv dimension
        num_kv_blocks = triton.cdiv(centroid_mask.shape[2], BLOCK_KV)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[1], num_kv_blocks)
        num_warps = 4 if Lq <= 64 else 8

        # create o_tmp to hold temporary outputs
        o_tmp = torch.zeros((q.shape[0], q.shape[1], num_kv_blocks, q.shape[2], q.shape[3]), device=q.device, dtype=torch.float32)
        L = torch.zeros((1, q.shape[0] * q.shape[1], num_kv_blocks, q.shape[2]), device=q.device, dtype=torch.float32)
        M = torch.zeros((1, q.shape[0] * q.shape[1], num_kv_blocks, q.shape[2]), device=q.device, dtype=torch.float32) - float('inf')


        # Centroid Replacement Kernel
        _fwd_centroid_kernel_attn[grid](
            v, qk,
            num_keys,
            sm_scale,
            o_tmp,
            centroid_mask,
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            centroid_mask.stride(0), centroid_mask.stride(1), centroid_mask.stride(2),
            q.shape[0], q.shape[1], q.shape[2], centroid_mask.shape[2],
            L, M,
            num_kv_blocks,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lq,
            BLOCK_KV=BLOCK_KV,
            num_warps=num_warps
        )

        # Reduction Kernel
        grid2 = (1, q.shape[1], 1)
        _reduce_kernel_attn[grid2](
            o_tmp, o,
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            L, M,
            l, m,
            q.shape[2], centroid_mask.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DMODEL=Lq,
            num_warps=num_warps
        )

        return o, m, l

centroid_lookup_kernel_simple = _centroid_lookup_gen.apply
qk_sparse_attention_kernel = _attention_qk_gen.apply
centroid_replacement_kernel = _centroid_replacement.apply

# wrapper function
def centroid_lookup(q, centroids, clabels, num_keys, sm_scale, global_threshold, num_key_value_groups, correction=False):
    return centroid_lookup_kernel_simple(q, centroids, clabels, num_keys, sm_scale, global_threshold, num_key_value_groups)

# wrapper function
def dynamic_sparse_attention(q, k, v, k_idx, head_kv_len, num_kv_blocks, head_start_block, out, max, denom, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return qk_sparse_attention_kernel(q, k, v, k_idx, out, max, denom, head_start_block, head_kv_len, num_kv_blocks, sm_scale)

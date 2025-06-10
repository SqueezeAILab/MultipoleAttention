import math
import numpy as np
import pytest
import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time
import pdb

### Start flash attention baseline
@triton.jit
def _fwd_kernel_qk_baseline(
    Q, K, V,
    sm_scale, Out,
    sqz, sqh, sqm, sqd,
    skz, skh, skn, skd,
    svz, svh, svn, svd,
    soz, soh, sokv, som, sod,
    L, M, num_kv_blocks,
    Z, H, N_CTX_Q, N_CTX_KV,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0).to(tl.int64)
    bh_id     = tl.program_id(1).to(tl.int64)
    kv_block_id = tl.program_id(2).to(tl.int64)
    batch_id = bh_id // H
    head_id  = bh_id %  H

    # offset into block (shift KV offsets by BLOCK_N * num_offsets)
    block_offset = (kv_block_id * BLOCK_KV).to(tl.int64)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64)
    offs_n = block_offset + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # qkv indices
    offs_q = ( batch_id * sqz + head_id  * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd )
    offs_k = ( batch_id * skz + head_id * skh + offs_n[None, :] * skn + offs_d[:, None]  * skd )
    offs_v = ( batch_id * svz + head_id * svh + offs_n[:, None] * svn + offs_d[None, :] * svd )

    # pointers to m and l
    m_prev = tl.zeros([16], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([16], dtype=tl.float32)
    acc = tl.zeros([16, BLOCK_DMODEL], dtype=tl.float32)

    # number of kv blocks to iterate over
    tmp = 1
    tmp = tmp.to(tl.int64)
    end_n = tmp * (BLOCK_KV // BLOCK_N)

    # need to trim depending on kvlen
    if N_CTX_KV < BLOCK_KV + block_offset:
        remaining_keys = (N_CTX_KV - block_offset)
        end_n = tmp * (remaining_keys + BLOCK_N - 1) // BLOCK_N

    # Load values
    q_vals = tl.load(Q + offs_q, mask=(offs_m[:, None] < N_CTX_Q) , other=0)

    # rescale sm_scale
    sm_scale *= 1.44269504  # 1/log(2)

    # loop over blocks
    for i in range(0, end_n):

        # Load values for K (use kv_len to detect last valid key)
        k_vals = tl.load(K + offs_k, mask=(offs_n[None, :] < N_CTX_KV) , other=0)

        # compute qk
        qk = tl.zeros([16, BLOCK_N], dtype=tl.bfloat16)
        qk += tl.dot(q_vals, k_vals)
        qk *= sm_scale

        # mask here
        qk_mask = (offs_n[None, :] < N_CTX_KV) & (offs_m[:, None] < N_CTX_Q)
        qk = tl.where(qk_mask, qk, float("-inf"))

        # compute attention weights - log2 version
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        qk = qk - m_curr[:, None]
        p = tl.math.exp2(qk)
        l_tmp = tl.sum(p, 1)
        alpha = tl.math.exp2(m_prev - m_curr)
        l_prev = l_prev * alpha
        l_curr = l_prev + l_tmp
        acc = acc * alpha[:, None]
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0).to(tl.float32)
        acc += tl.dot(p, v_vals)

        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr

        # update offsets
        offs_n += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_v += BLOCK_N * svn

    # epilogue
    acc = acc / l_prev[:, None]

    # guard against 0-denom
    nan_mask = acc != acc
    acc = tl.where(nan_mask, 0, acc)

    # store L and M
    offs_L = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + kv_block_id * N_CTX_Q + offs_m
    offs_M = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + kv_block_id * N_CTX_Q + offs_m
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)

    # store results to output
    offs_o = batch_id * soz + head_id * soh + kv_block_id * sokv + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=(offs_m[:, None] < N_CTX_Q))



@triton.jit
def _reduce_kernel_qk_baseline(
        Out_tmp, Out,
        sotz, soth, sotkv, sotm, sotd,
        soz, soh, som, sod,
        L, M,
        H, N_CTX_Q, N_CTX_KV,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr
    ):

    start_m = tl.program_id(0).to(tl.int64)
    bh_id     = tl.program_id(1).to(tl.int64)
    batch_id = bh_id // H
    head_id  = bh_id %  H

    # number of output blocks to loop over
    num_o_tmp_blocks = (N_CTX_KV.to(tl.int64) + BLOCK_KV - 1) // BLOCK_KV
    num_o_tmp_blocks = num_o_tmp_blocks.to(tl.int64)

    # get query start indices
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # initialize offsets here
    offs_lm = batch_id * H * num_o_tmp_blocks * N_CTX_Q + head_id * num_o_tmp_blocks * N_CTX_Q + offs_m
    offs_o_tmp = batch_id * sotz + head_id * soth + offs_m[:, None] * sotm + offs_d[None, :] * sotd
    offs_o = batch_id * soz + head_id * soh + offs_m[:, None] * som + offs_d[None, :] * sod

    # pointers to m and l - use M and L from causal step as input
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # loop over blocks
    for _ in range(0, num_o_tmp_blocks):

        # load out_tmp block
        o_tmp_vals = tl.load(Out_tmp + offs_o_tmp, mask=offs_m[:, None] < N_CTX_Q, other=0)

        # Load current L / M
        m_curr = tl.load(M + offs_lm, mask=offs_m < N_CTX_Q, other=0)
        l_curr = tl.load(L + offs_lm, mask=offs_m < N_CTX_Q, other=0)

        # scale tiles up by denom before adjusting
        acc *= l_prev[:, None]
        o_tmp_vals *= l_curr[:, None]

        # compute largest value from both new block and accum
        m_tmp = tl.maximum(m_curr, m_prev)

        # amount to shift by for M1 / M2
        shift1 = tl.math.exp2(m_prev - m_tmp)
        shift2 = tl.math.exp2(m_curr - m_tmp)

        # adjust denominators using largest value from both new block and accum
        l_prev *= shift1
        l_curr *= shift2

        # rescale acc and o_tmp_vals
        acc *= shift1[:, None]
        o_tmp_vals *= shift2[:, None]

        # accumulate
        acc += o_tmp_vals

        # update m_i and l_i
        l_prev += l_curr
        m_prev = m_tmp

        # rescale acc
        acc /= l_prev[:, None]

        # update offsets
        offs_o_tmp += sotkv
        offs_lm += N_CTX_Q

    # store out block
    tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)

class _attention_qk_baseline(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK_N = 64
        BLOCK_M = 64
        BLOCK_KV = BLOCK_KV_GLOBAL

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

        num_kv_blocks = triton.cdiv(k.shape[2], BLOCK_KV)
        grid = (1, q.shape[0] * q.shape[1], num_kv_blocks)
        num_warps = 4 if Lk <= 64 else 8

        # create o_tmp to hold temporary outputs
        o_tmp = torch.empty((q.shape[0], q.shape[1], num_kv_blocks, q.shape[2], q.shape[3]), device=q.device, dtype=torch.bfloat16)

        # L - running sum, M - running max
        L = torch.empty((q.shape[0], q.shape[1], num_kv_blocks, q.shape[2]), device=q.device, dtype=torch.float32)
        M = torch.empty((q.shape[0], q.shape[1], num_kv_blocks, q.shape[2]), device=q.device, dtype=torch.float32)

        _fwd_kernel_qk_baseline[grid](
            q, k, v,
            sm_scale, o_tmp,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            L, M, num_kv_blocks,
            q.shape[0], q.shape[1], q.shape[2], k.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        # Reduction Kernel
        grid2 = (1,q.shape[0] * q.shape[1], 1)
        o = torch.empty_like(q)

        _reduce_kernel_qk_baseline[grid2](
            o_tmp, o,
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            L, M,
            q.shape[1], q.shape[2], k.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o


@triton.jit
def _fwd_centroid_kernel_qk_gen_splitkv(
    Q, K, QK,
    NKeys,
    CLabels,
    sm_scale,
    Score,
    threshold,
    sqz, sqh, sqm, sqd,
    skz, skh, skn, skd,
    ssz, ssh, ssn, ssq,
    sqkz, sqkh, sqkm, sqkn,
    Z, H, N_CTX_Q, N_CTX_KV, N_CENTROIDS,
    L, M,
    num_kv_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    start_m = tl.program_id(0).to(tl.int64)
    bh_id   = tl.program_id(1).to(tl.int64)
    kv_split_id = tl.program_id(2).to(tl.int64)
    head_id   = bh_id %  H
    batch_id  = bh_id // H

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64)
    block_offset = kv_split_id * BLOCK_KV
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64) + block_offset
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # indices for qk
    offs_q = batch_id * sqz + head_id  * sqh  + offs_m[:, None] * sqm  + offs_d[None, :] * sqd
    offs_k = batch_id * skz + head_id * skh + offs_n[None, :] * skn + offs_d[:, None] * skd

    # num‑keys / centroid labels
    offs_nk = batch_id * N_CENTROIDS * H + head_id * N_CENTROIDS + offs_n

    # indices for qk res
    offs_qk = batch_id * sqkz + head_id * sqkh + offs_n[None, :] * sqkn + offs_m[:, None] * sqkm

    # need to trim depending on N_CENTROIDS
    tmp = 1
    tmp = tmp.to(tl.int64)
    end_n = tmp * (BLOCK_KV // BLOCK_N)

    # need to trim depending on KVLEN
    if N_CENTROIDS < BLOCK_KV + block_offset:
        remaining_keys = (N_CENTROIDS - block_offset)
        end_n = (remaining_keys + BLOCK_N - 1) // BLOCK_N

    # Load queries (pad to 16x128)
    q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)

    # store estimate denominator
    m_prev = tl.zeros([16], dtype=tl.float32) - float("inf")
    denom = tl.zeros([16], dtype=tl.float32)
    sm_scale *= 1.44269504 # 1/log(2)

    # offset for max output score
    offs_s = batch_id * ssz + head_id * ssh + offs_n[None, :] * ssn + offs_m[:, None] * ssq

    # loop over blocks
    for _ in range(0, end_n):

        # Load values for K (use kv_len to detect last valid key)
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CENTROIDS, other=0)

        # load number of keys per centroids here
        nkeys = tl.load(NKeys + offs_nk, mask=offs_n < N_CENTROIDS, other=0)

        # compute qk
        qk = tl.zeros([16, BLOCK_N], dtype=tl.bfloat16)
        qk += tl.dot(q_vals, k_vals)
        qk *= sm_scale

        # mask out beyond seqlen
        qk = tl.where(offs_n[None, :] < N_CENTROIDS, qk, float("-inf"))
        storemask = (offs_n[None, :] < N_CENTROIDS) & (offs_m[:, None] < N_CTX_Q)

        # save qk
        tl.store(QK + offs_qk, qk, mask=storemask)

        # normalization for numerical stability
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        p_store = tl.math.exp2(qk)
        qk = qk - m_curr[:, None]
        p = tl.math.exp2(qk)

        # compute exp_qk * num_c, then sum
        p_tmp = p * nkeys[None, :]
        l_tmp = tl.sum(p_tmp, 1)
        alpha = tl.math.exp2(m_prev - m_curr)
        denom = denom * alpha + l_tmp

        # update m_prev
        m_prev = m_curr

        # store qk value
        tl.store(Score + offs_s, p_store, mask=storemask)

        # update offsets
        offs_n += BLOCK_N
        offs_nk += BLOCK_N
        offs_qk += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_s += BLOCK_N * ssn

    # store L here
    offs_L = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + offs_m * num_kv_blocks + kv_split_id
    offs_M = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + offs_m * num_kv_blocks + kv_split_id

    tl.store(L + offs_L, denom, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)

@torch.compile
def topk_option1_fused(M, L, score, centroid_labels, num_tokens_retained):
    maxval = M.amax(dim=-1, keepdim=True)
    shift  = torch.exp2(M - maxval)
    L      = L * shift
    Lsum   = L.sum(dim=-1, keepdim=True)
    avg_score    = score.mean(dim=-1)

    # gather per-token scores and cast to half
    avg_score_per_token = torch.gather(avg_score, 2, centroid_labels).half()
    # unsorted top-k
    top_vals, top_idx = torch.topk(
        avg_score_per_token,
        int(num_tokens_retained),
        dim=-1,
        largest=True,
        sorted=False
    )

    # mask out the “partial” cluster in the event of ties at the threshold
    thresh, _ = top_vals.min(dim=-1, keepdim=True)
    centroid_mask = avg_score > thresh
    is_min = top_vals == thresh
    top_idx = top_idx.masked_fill(is_min, -1)
    return top_idx, centroid_mask

class _centroid_lookup_simple(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, centroid_labels, num_keys, sm_scale, threshold, num_key_value_groups):
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
        BLOCK_KV = BLOCK_KV_GLOBAL

        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]
        assert Lq == Lk
        assert Lk in {64, 128}

        # number of KV heads and GQA factor
        NUM_HEADS = q.shape[1] * q.shape[2]
        NUM_KV_HEADS = k.shape[1]
        GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS

        assert(q.shape[1] == NUM_KV_HEADS)
        assert(q.shape[2] == GQA_FACTOR)

        # set up grid - need shape to be ( ceil(q_seqlen / BLOCKSIZE) , NUMHEADS * ceil(nonzero_kv_seqlen_per_head / BLOCKSIZE) )
        # gives ( ceil(q_seqlen / BLOCKSIZE) , NUM_NONZERO_KV_BLOCKS)
        num_warps = 4 if Lk <= 64 else 8

        # split along kv dimension
        num_kv_blocks = triton.cdiv(k.shape[2], BLOCK_KV)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0]*q.shape[1], num_kv_blocks)

        # L - running sum, M - running max
        num_query_blocks = triton.cdiv(q.shape[2], BLOCK_M)
        score = torch.empty((q.shape[0], q.shape[1], k.shape[2], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((1, q.shape[0] * q.shape[1], q.shape[2], num_kv_blocks), device=q.device, dtype=torch.float32)
        M = torch.empty((1, q.shape[0] * q.shape[1], q.shape[2], num_kv_blocks), device=q.device, dtype=torch.float32) - float('inf')

        qk = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=torch.bfloat16)

        _fwd_centroid_kernel_qk_gen_splitkv[grid](
            q, k, qk,
            num_keys, centroid_labels,
            sm_scale,
            score,
            threshold,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            score.stride(0), score.stride(1), score.stride(2), score.stride(3),
            qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
            q.shape[0], q.shape[1], q.shape[2], centroid_labels.shape[2], k.shape[2],
            L, M,
            num_kv_blocks,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            BLOCK_KV=BLOCK_KV,
            num_warps=num_warps
        )

        B, H, C, _ = score.shape
        N = centroid_labels.shape[-1]
        num_tokens_retained = (centroid_labels.shape[-1] * SPARSITY_RATIO)

        top_idx, centroid_mask = topk_option1_fused(M, L, score, centroid_labels, num_tokens_retained)
        return top_idx, centroid_mask, qk


### Begin replacement kernels
@triton.jit
def _fwd_centroid_kernel_attn(
    Q, K, V, QK,
    NKeys,
    sm_scale,
    Out,
    Mask,
    sqz, sqh, sqm, sqd,
    skz, skh, skn, skd,
    svz, svh, svn, svd,
    sqkz, sqkh, sqkm, sqkn,
    soz, soh, sokv, som, sod,
    smz, smh, smn,
    Z, H, N_CTX_Q, N_CTX_KV, N_CENTROIDS,
    L, M,
    num_kv_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    start_m = tl.program_id(0).to(tl.int64)
    bh_id     = tl.program_id(1).to(tl.int64)
    kv_split_id = tl.program_id(2).to(tl.int64)
    head_id   = bh_id % H
    batch_id  = bh_id // H

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64)
    block_offset = kv_split_id * BLOCK_KV
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64) + block_offset
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # indices for QKV
    offs_qk = batch_id * sqkz + head_id * sqkh + offs_n[None, :] * sqkn + offs_m[:, None] * sqkm
    offs_v  = batch_id * svz  + head_id * svh  + offs_n[:, None] * svn   + offs_d[None, :] * svd
    offs_msk= batch_id * smz  + head_id * smh  + offs_n * smn

    # indices for num keys per centroid
    offs_nk = head_id * N_CENTROIDS + offs_n

    # need to trim depending on N_CENTROIDS
    tmp = 1
    tmp = tmp.to(tl.int64)
    end_n = tmp * (BLOCK_KV // BLOCK_N)

    # need to trim depending on KVLEN
    if N_CENTROIDS < BLOCK_KV + block_offset:
        remaining_keys = (N_CENTROIDS - block_offset)
        end_n = (remaining_keys + BLOCK_N - 1) // BLOCK_N

    # pointers to m and l
    m_prev = tl.zeros([16], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([16], dtype=tl.float32)
    acc = tl.zeros([16, BLOCK_DMODEL], dtype=tl.float32)

    # rescale sm_scale
    sm_scale *= 1.44269504  # 1/log(2)

    # loop over blocks
    for _ in range(0, end_n):

        # load number of keys per centroids here
        nkeys = tl.load(NKeys + offs_nk, mask=offs_n < N_CENTROIDS, other=0)
        mask = tl.load(Mask + offs_msk, mask=offs_n < N_CENTROIDS, other=1)

        # here, we assume mask == 0 is replace
        active = (mask == 0) & (nkeys > 0)

        if tl.max(active) == 1:
            qk = tl.load(QK + offs_qk, mask=(offs_n[None, :] < N_CENTROIDS) & (offs_m[:, None] < N_CTX_Q), other=float('-inf'))
            qk = tl.where(active[None, :], qk, float("-inf"))

            # normalization for numerical stability
            m_curr = tl.maximum(tl.max(qk, 1), m_prev)
            qk = qk - m_curr[:, None]
            p = tl.math.exp2(qk)

            # compute exp_qk * num_c, then sum
            p_tmp = p * nkeys[None, :] # masking is implictly handled here
            l_tmp = tl.sum(p_tmp, 1)
            alpha = tl.math.exp2(m_prev - m_curr)
            l_prev = l_prev * alpha
            l_curr = l_prev + l_tmp
            acc = acc * alpha[:, None]

            # update acc
            v_vals = tl.load(V + offs_v, mask=(offs_n[:, None] < N_CENTROIDS) , other=0).to(tl.float32)
            acc += tl.dot(p_tmp, v_vals)

            # update m_i and l_i
            l_prev = l_curr
            m_prev = m_curr

            # update offsets
            offs_n += BLOCK_N
            offs_nk += BLOCK_N
            offs_qk += BLOCK_N
            offs_v += BLOCK_N * svn
            offs_msk += BLOCK_N * smn
        else:
            offs_n += BLOCK_N
            offs_nk += BLOCK_N
            offs_qk += BLOCK_N
            offs_v += BLOCK_N * svn
            offs_msk += BLOCK_N * smn

    # epilogue
    acc = acc / l_prev[:, None]

    # guard against 0-denom
    nan_mask = acc != acc
    acc = tl.where(nan_mask, 0, acc)

    # store L here
    offs_L = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + kv_split_id * N_CTX_Q + offs_m
    offs_M = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + kv_split_id * N_CTX_Q + offs_m
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)

    # store results to output
    offs_o  = batch_id * soz + head_id * soh + kv_split_id * sokv + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=(offs_m[:, None] < N_CTX_Q))


@triton.jit
def _reduce_kernel_attn(
        Out_tmp, Out,
        sotz, soth, sotkv, sotm, sotd,
        soz, soh, som, sod,
        L, M,
        H, N_CTX_Q, N_CTX_KV,
        BLOCK_M: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr
    ):
    start_m = tl.program_id(0).to(tl.int64)
    bh_id   = tl.program_id(1).to(tl.int64)
    head_id   = bh_id %  H
    batch_id  = bh_id // H

    # number of output blocks to loop over
    num_o_tmp_blocks = (N_CTX_KV.to(tl.int64) + BLOCK_KV - 1) // BLOCK_KV
    num_o_tmp_blocks = num_o_tmp_blocks.to(tl.int64)

    # get query start indices
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # initialize offsets here
    offs_o_tmp = batch_id * sotz + head_id * soth + offs_m[:, None] * sotm + offs_d[None, :] * sotd
    offs_o = batch_id * soz + head_id * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    offs_lm = batch_id * H * num_o_tmp_blocks * N_CTX_Q + head_id * num_o_tmp_blocks * N_CTX_Q + offs_m

    offs_lm = offs_lm.to(tl.int64)
    offs_o_tmp = offs_o_tmp.to(tl.int64)
    offs_o = offs_o.to(tl.int64)

    # initialize from scratch
    m_prev = tl.zeros([16], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([16], dtype=tl.float32)
    acc = tl.zeros([16, BLOCK_DMODEL], dtype=tl.float32)

    # loop over blocks
    for _ in range(0, num_o_tmp_blocks):

        # load out_tmp block
        o_tmp_vals = tl.load(Out_tmp + offs_o_tmp, mask=(offs_m[:, None] < N_CTX_Q), other=0).to(tl.float32)

        # Load current L / M
        m_curr = tl.load(M + offs_lm, mask=offs_m < N_CTX_Q, other=-float("inf"))
        l_curr = tl.load(L + offs_lm, mask=offs_m < N_CTX_Q, other=0)

        # scale tiles up by denom before adjusting
        acc *= l_prev[:, None]
        o_tmp_vals *= l_curr[:, None]

        # compute largest value from both new block and accum
        m_tmp = tl.maximum(m_curr, m_prev)

        # amount to shift by for M1 / M2
        shift1 = tl.math.exp2(m_prev - m_tmp)
        shift2 = tl.math.exp2(m_curr - m_tmp)

        # adjust denominators using largest value from both new block and accum
        l_prev *= shift1
        l_curr *= shift2

        # rescale acc and o_tmp_vals
        acc *= shift1[:, None]
        o_tmp_vals *= shift2[:, None]

        # accumulate
        acc += o_tmp_vals

        # update m_i and l_i
        l_prev += l_curr
        m_prev = m_tmp

        # rescale acc
        acc /= l_prev[:, None]

        # guard against 0-denom
        nan_mask = acc != acc
        acc = tl.where(nan_mask, 0, acc)

        # update offsets
        offs_o_tmp += sotkv
        offs_lm += N_CTX_Q

    # store out block
    tl.store(Out + offs_o, acc, mask=(offs_m[:, None] < N_CTX_Q))

class _centroid_replacement(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, centroid_labels, centroid_mask, num_keys, sm_scale, threshold, qk):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK_N = 64
        BLOCK_M = 64
        BLOCK_KV = BLOCK_KV_GLOBAL

        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]
        assert Lq == Lk
        assert Lk in {64, 128}
        # split along kv dimension

        # FIXME: Refactor so this does it with calculated indices.
        num_kv_blocks = triton.cdiv(k.shape[2], BLOCK_KV)

        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], num_kv_blocks)
        num_warps = 4 if Lk <= 64 else 8

        # create o_tmp to hold temporary outputs
        o_tmp = torch.zeros((q.shape[0], q.shape[1], num_kv_blocks, q.shape[2], q.shape[3]), device=q.device, dtype=torch.float32)

        L = torch.zeros((1, q.shape[0] * q.shape[1], num_kv_blocks, q.shape[2]), device=q.device, dtype=torch.float32)
        M = torch.zeros((1, q.shape[0] * q.shape[1], num_kv_blocks, q.shape[2]), device=q.device, dtype=torch.float32) - float('inf')

        # This should be almost equivalent to full xattn
        _fwd_centroid_kernel_attn[grid](
            q, k, v, qk,
            num_keys,
            sm_scale,
            o_tmp,
            centroid_mask,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            centroid_mask.stride(0), centroid_mask.stride(1), centroid_mask.stride(2),
            q.shape[0], q.shape[1], q.shape[2], centroid_labels.shape[2], k.shape[2],
            L, M,
            num_kv_blocks,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            BLOCK_KV=BLOCK_KV,
            num_warps=num_warps
        )

        # Reduction Kernel
        grid2 = (1, q.shape[0] * q.shape[1], 1)
        o = torch.empty_like(q)

        _reduce_kernel_attn[grid2](
            o_tmp, o,
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            L, M,
            q.shape[1], q.shape[2], k.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        return o

### Begin sparse attention
# qk kernel (sparse balanced) - revised kernel
@triton.jit
def _fwd_kernel_qk_gen(
    Q, K, V, Kidx,
    sm_scale, Out,
    sqz, sqh, sqm, sqd,
    skz, skh, skn, skd,
    svz, svh, svn, svd,
    skiz, skih, skin,
    soz, soh, sokv, som, sod,
    KV_BUDGET,
    L, M,
    Z, H, N_CTX_Q,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0).to(tl.int64)
    bh_id   = tl.program_id(1)
    kv_blk  = tl.program_id(2)

    batch_id = bh_id // H
    head_id  = bh_id %  H

    # initialize KV block parameters
    block_offset = kv_blk * BLOCK_KV
    kv_len = KV_BUDGET

    # number of tiles
    kv_block_len  = tl.minimum(KV_BUDGET - block_offset, BLOCK_KV)
    num_tiles = (kv_block_len + BLOCK_N - 1) // BLOCK_N
    num_kv_blocks = (kv_len + BLOCK_KV - 1) // BLOCK_KV

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64)
    offs_n = block_offset + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # qk indices
    offs_q = batch_id * sqz + head_id * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    offs_kidx = batch_id * skiz + head_id * skih + offs_n * skin
    offs_k_base = batch_id * skz + head_id * skh + offs_d[:, None] * skd
    offs_v_base = batch_id * svz + head_id * svh + offs_d[None, :] * svd

    # pointers to m and l
    m_prev = tl.zeros([16], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([16], dtype=tl.float32)
    acc = tl.zeros([16, BLOCK_DMODEL], dtype=tl.float32)

    # Load values
    q_vals = tl.load(Q + offs_q, mask=(offs_m[:, None] < N_CTX_Q) , other=0)

    # rescale sm_scale
    sm_scale *= 1.44269504

    # cast to int64
    offs_k_base = offs_k_base.to(tl.int64)
    offs_v_base = offs_v_base.to(tl.int64)
    svn = svn.to(tl.int64)
    offs_n = offs_n.to(tl.int64)
    kv_len = kv_len.to(tl.int64)

    # workaround for compiler error
    k_idx_vals = tl.zeros([BLOCK_N], dtype=tl.int64)

    # loop over blocks
    for i in range(0, num_tiles):

        if i>0:
            offs_n += BLOCK_N

        # load k_idx here using offs_k
        k_idx_vals = tl.load(Kidx + offs_kidx, mask=offs_n < kv_len, other = 0).to(tl.int64)

        # cast to 1 bit to get bool mask
        zero  = tl.zeros_like(k_idx_vals)
        kv_valid_1d = (k_idx_vals >= zero).to(tl.int1)
        k_idx_vals_safe  = tl.zeros_like(k_idx_vals) - 1
        k_idx_vals_safe = k_idx_vals_safe + k_idx_vals

        # mask for k and v
        k_mask = kv_valid_1d[None, :] & (offs_n[None, :] < kv_len)
        v_mask = kv_valid_1d[:, None] & (offs_n[:, None] < kv_len)

        # compute K/V addresses to load
        offs_k = offs_k_base + k_idx_vals[None, :] * skn
        offs_v = offs_v_base + k_idx_vals[:, None] * svn

        # Load values for K (use kv_len to detect last valid key)
        k_vals = tl.load(K + offs_k, mask=k_mask, other=0)

        # compute qk
        qk = tl.dot(q_vals, k_vals)
        qk *= sm_scale

        # mask here
        qk = tl.where(k_mask, qk, float("-inf"))

        # compute attention weights - log2 version
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        qk = qk - m_curr[:, None]
        p = tl.math.exp2(qk)
        l_tmp = tl.sum(p, 1)
        alpha = tl.math.exp2(m_prev - m_curr)
        l_prev = l_prev * alpha
        l_curr = l_prev + l_tmp
        acc = acc * alpha[:, None]

        # update acc
        p = p.to(Q.dtype.element_ty)
        v_vals = tl.load(V + offs_v, mask=v_mask, other=0)
        acc += tl.dot(p, v_vals)

        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr

        # update offsets
        offs_kidx += BLOCK_N * skin

    # epilogue
    acc = acc / l_prev[:, None]

    # reset offs_m here
    offs_L = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + kv_blk * N_CTX_Q + offs_m
    offs_M = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + kv_blk * N_CTX_Q + offs_m

    # store L and M
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)

    # store results to output
    offs_o = batch_id * soz + head_id * soh + kv_blk * sokv + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=(offs_m[:, None] < N_CTX_Q))


@triton.jit
def _reduce_kernel_qk_gen(
        Out_tmp, Out,
        sotz, soth, sotkv, sotm, sotd,
        soz, soh, som, sod,
        L, M,
        Lout, Mout,
        KV_BUDGET,
        B, H, N_CTX_Q,
        BLOCK_M: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr
    ):
    start_m = tl.program_id(0).to(tl.int64)
    bh_id   = tl.program_id(1)
    batch_id = bh_id // H
    head_id  = bh_id %  H

    # initialize KV block parameters
    num_kv_blocks = (KV_BUDGET + BLOCK_KV - 1) // BLOCK_KV

    # get query start indices
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # initialize offsets here
    offs_lm = batch_id * H * num_kv_blocks * N_CTX_Q + head_id * num_kv_blocks * N_CTX_Q + offs_m
    offs_o_tmp = batch_id * sotz + head_id  * soth + offs_m[:, None] * sotm + offs_d[None, :] * sotd
    offs_o = batch_id * soz + head_id  * soh + offs_m[:, None] * som + offs_d[None, :] * sod

    # pointers to m and l - use M and L from causal step as input
    offs_M = batch_id * H * N_CTX_Q + head_id  * N_CTX_Q + offs_m
    offs_L = batch_id * H * N_CTX_Q + head_id  * N_CTX_Q + offs_m
    m_prev = tl.load(Mout + offs_M, mask=(offs_m < N_CTX_Q), other=float("-inf")).to(tl.float32)
    l_prev = tl.load(Lout + offs_L, mask=(offs_m < N_CTX_Q), other=0).to(tl.float32)
    acc = tl.load(Out + offs_o, mask=(offs_m[:, None] < N_CTX_Q), other=0).to(tl.float32)

    # loop over blocks
    for _ in range(0, num_kv_blocks):

        # load out_tmp block
        o_tmp_vals = tl.load(Out_tmp + offs_o_tmp, mask=offs_m[:, None] < N_CTX_Q, other=0)

        # Load current L / M
        m_curr = tl.load(M + offs_lm, mask=offs_m < N_CTX_Q, other=float("-inf"))
        l_curr = tl.load(L + offs_lm, mask=offs_m < N_CTX_Q, other=0)

        # scale tiles up by denom before adjusting
        acc *= l_prev[:, None]
        o_tmp_vals *= l_curr[:, None]

        # compute largest value from both new block and accum
        m_tmp = tl.maximum(m_curr, m_prev)

        # amount to shift by for M1 / M2
        shift1 = tl.math.exp2(m_prev - m_tmp)
        shift2 = tl.math.exp2(m_curr - m_tmp)

        # adjust denominators using largest value from both new block and accum
        l_prev *= shift1
        l_curr *= shift2

        # rescale acc and o_tmp_vals
        acc *= shift1[:, None]
        o_tmp_vals *= shift2[:, None]

        # accumulate
        acc += o_tmp_vals

        # update m_i and l_i
        l_prev += l_curr
        m_prev = m_tmp

        # rescale acc
        acc /= l_prev[:, None]

        # guard against 0-denom
        nan_mask = acc != acc
        acc = tl.where(nan_mask, 0, acc)

        # update offsets
        offs_o_tmp += sotkv
        offs_lm += N_CTX_Q

    # store out block
    tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)

    # store Lout and Mout
    tl.store(Lout + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(Mout + offs_M, m_prev, mask=offs_m < N_CTX_Q)


class _attention_qk_gen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, k_idx, Out, Mout, Lout, sm_scale):

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
        BLOCK_KV = BLOCK_KV_GLOBAL

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {64, 128}

        # num_kv_heads
        NUM_HEADS = q.shape[1] * q.shape[2]
        NUM_KV_HEADS = k.shape[1]
        GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS
        KV_BUDGET = k_idx.shape[-1]
        num_kv_blocks = math.ceil(KV_BUDGET / BLOCK_KV)

        assert(q.shape[1] == NUM_KV_HEADS)
        assert(q.shape[2] == GQA_FACTOR)

        # set up grid - need shape to be ( ceil(q_seqlen / BLOCKSIZE) , NUMHEADS * ceil(nonzero_kv_seqlen_per_head / BLOCKSIZE) )
        # gives ( ceil(q_seqlen / BLOCKSIZE) , NUM_NONZERO_KV_BLOCKS)
        grid = (1, q.shape[0] * q.shape[1], num_kv_blocks)
        num_warps = 4 if Lk <= 64 else 8

        # create o_tmp to hold temporary outputs
        o_tmp = torch.empty((q.shape[0], q.shape[1], num_kv_blocks, GQA_FACTOR, q.shape[3]), device=q.device, dtype=torch.bfloat16)

        # L - running sum, M - running max
        L = torch.empty((q.shape[0], q.shape[1], num_kv_blocks, GQA_FACTOR), device=q.device, dtype=torch.float32)
        M = torch.empty((q.shape[0], q.shape[1], num_kv_blocks, GQA_FACTOR), device=q.device, dtype=torch.float32)

        _fwd_kernel_qk_gen[grid](
            q, k, v, k_idx,
            sm_scale, o_tmp,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            k_idx.stride(0), k_idx.stride(1), k_idx.stride(2),
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            KV_BUDGET,
            L, M,
            q.shape[0], q.shape[1], q.shape[2],
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
            o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2), o_tmp.stride(3), o_tmp.stride(4),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            L, M,
            Lout, Mout,
            KV_BUDGET,
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        return Out, Mout, Lout


flash_attention_optimized = _attention_qk_baseline.apply
centroid_lookup_kernel = _centroid_lookup_simple.apply
centroid_replacement_kernel = _centroid_replacement.apply
sparse_flash_attn = _attention_qk_gen.apply

### Begin testing suite
def get_tensors(BATCH, H, N_CTX, D_HEAD):

    q = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device="cuda", requires_grad=True)

    return q, k, v

def make_centroid_labels(B, H, N_CTX_KV, N_CENTROIDS, device):
    return torch.randint(
        0, N_CENTROIDS, (B, H, N_CTX_KV),
        dtype=torch.int64, device=device
    )

def count_keys_per_centroid(labels: torch.Tensor,
                            N_CENTROIDS: int) -> torch.Tensor:
    B, H, _ = labels.shape
    out  = torch.zeros(B, H, N_CENTROIDS,
                       dtype=torch.int32, device=labels.device)
    ones = torch.ones_like(labels, dtype=torch.int32)
    out.scatter_add_(2, labels, ones)
    return out

def test_flash_decoding():
    q, _, _ = get_tensors(BATCH, N_CTX_Q, H_Q, D_HEAD)
    _, k, v = get_tensors(BATCH, N_CTX_KV, H_KV, D_HEAD)
    sm_scale = 1.0 / math.sqrt(q.size(-1))

    assert(q.shape[2] == 1)
    num_kv_groups = H_Q // H_KV
    q = q.reshape(BATCH, H_KV, num_kv_groups, D_HEAD)

    fn = lambda: flash_attention_optimized(q, k, v, sm_scale)
    return triton.testing.do_bench(fn, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])

def test_centroid_lookup():
    q, _, _ = get_tensors(BATCH, N_CTX_Q, H_Q, D_HEAD)
    _, k, _ = get_tensors(BATCH, N_CENTROIDS, H_KV, D_HEAD)
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    centroid_labels = make_centroid_labels(
        BATCH, H_KV, N_CTX_KV, N_CENTROIDS, q.device
    )
    num_keys = count_keys_per_centroid(centroid_labels, N_CENTROIDS)

    q = q.view(BATCH, H_KV, H_Q // H_KV, D_HEAD)

    fn = lambda: centroid_lookup_kernel(
        q, k, centroid_labels, num_keys, sm_scale, None, 4
    )
    return triton.testing.do_bench(
        fn, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8]
    )

def test_centroid_replacement():
    q, _, _ = get_tensors(BATCH, N_CTX_Q, H_Q, D_HEAD)
    _, k, v = get_tensors(BATCH, N_CENTROIDS, H_KV, D_HEAD)
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    centroid_labels = make_centroid_labels(
        BATCH, H_KV, N_CTX_KV, N_CENTROIDS, q.device
    )
    num_keys = count_keys_per_centroid(centroid_labels, N_CENTROIDS)

    q = q.view(BATCH, H_KV, H_Q // H_KV, D_HEAD)
    k_idx, centroid_mask, qk = centroid_lookup_kernel(q, k, centroid_labels, num_keys, sm_scale, None, 4)

    fn = lambda: centroid_replacement_kernel(q, k, v, centroid_labels, centroid_mask, num_keys, sm_scale, None, qk)
    return triton.testing.do_bench(fn, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])

def test_sparse_attn():

    q, _, _ = get_tensors(BATCH, N_CTX_Q, H_Q, D_HEAD)
    _, k_centr, v_centr = get_tensors(BATCH, N_CENTROIDS, H_KV, D_HEAD)
    sm_scale = 1.0 / math.sqrt(q.size(-1))

    centroid_labels = make_centroid_labels(
        BATCH, H_KV, N_CTX_KV, N_CENTROIDS, q.device
    )
    num_keys = count_keys_per_centroid(centroid_labels, N_CENTROIDS)

    q = q.reshape(BATCH,H_KV,H_Q // H_KV, -1)

    k_idx, centroid_mask, qk = centroid_lookup_kernel(q, k_centr, centroid_labels, num_keys, sm_scale, None, 4)

    # Dummy inputs for merging
    o = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    _, k, v = get_tensors(BATCH, N_CTX_KV, H_KV, D_HEAD)

    fn = lambda: sparse_flash_attn(q, k, v, k_idx, o, M, L, sm_scale)
    return triton.testing.do_bench(fn, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])


BLOCK_KV_GLOBAL = 512
SPARSITY_RATIO = 0.1
N_CTX_Q, N_CTX_KV = 1, 1024 * 128 # 1024 * 512
BATCH, H_Q, H_KV, D_HEAD = 1, 32, 8, 128
N_CENTROIDS = int(0.0625*N_CTX_KV)

BATCHES = [1, 4, 16]

### Iterate over batch sizes for 128K context length and print runtime breakdown ###
for B in BATCHES:
    print(f'Batch Size: {B}, N_CTX_KV: {N_CTX_KV}')
    BATCH = B

    _, median, _ = test_flash_decoding()
    print(median)

    _, median, _ = test_centroid_lookup()
    print(median)

    _, median, _ = test_centroid_replacement()
    print(median)

    SPARSITY_RATIO = 0.05
    _, median, _ = test_sparse_attn()
    print(median)

    SPARSITY_RATIO = 0.1
    _, median, _ = test_sparse_attn()
    print(median)

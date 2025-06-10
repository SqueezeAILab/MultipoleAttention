import math
import numpy as np
import pytest
import torch
import triton
import triton.language as tl
import time

# qk kernel (sparse balanced) - revised kernel
@triton.jit
def _fwd_kernel_qk_gen(
    Q, K, V, Kidx,
    sm_scale, Out,
    sqz, sqh, sqm, sqd, # shape = (Z,H,N_CTX_Q,D)
    skz, skh, skn, skd, # shape = (Z,H,N_CTX_KV,D)
    svz, svh, svn, svd, # shape = (Z,H,N_CTX_KV,D)
    skiz, skih, skin, # shape = (Z,H,N_CTX_KV_VALID)
    soz, sokv, som, sod, # shape = (Z,N_KV_BLOCKS,N_CTX_Q,D)
    head_start_block, head_kv_len,
    L, M,
    Z, H, N_CTX_Q, N_CTX_KV, N_CTX_KV_VALID, NUM_KV_HEADS,
    NUMHEADS: tl.constexpr,
    BLOCK_M: tl.constexpr, # will load BLOCK_M queries
    BLOCK_N: tl.constexpr, # will compute self attention by blocks of BLOCK_N keys
    BLOCK_KV: tl.constexpr, # will compute this width of keys
    BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0).to(tl.int64) # idx of sequence length chunk of size 128 (BLOCK_N)
    kv_block_id = tl.program_id(1).to(tl.int64) # idx of KV block
    kv_block_id2 = kv_block_id.to(tl.int64)

    # new method - compute kv_head_id
    head_group_id = -1
    for i in range(NUMHEADS):
        head_start = tl.load(head_start_block + i).to(tl.int32)
        is_in_head = kv_block_id2 >= head_start
        head_group_id += is_in_head.to(tl.int32)

    # initialize KV block parameters
    head_start = tl.load(head_start_block + head_group_id)#.to(tl.int64) # kv block that head starts at
    kv_len = tl.load(head_kv_len + head_group_id) #.to(tl.int64) # num valid kv for current head

    # offset into block (shift KV offsets by BLOCK_N * num_offsets)
    block_offset = ((kv_block_id - head_start) * BLOCK_KV)#.to(tl.int64)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64) # indices of queries we want to process
    offs_n = block_offset + tl.arange(0, BLOCK_N).to(tl.int64) # indices of key indices we want to process, we start from [block_offset, block_offset+BLOCK_N-1] and update in the loop
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # query indices
    offs_q = head_group_id * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd

    # key index indices
    offs_kidx = head_group_id * skih + offs_n * skin

    # key / value indices
    offs_k_base = head_group_id * skh + offs_d[:, None] * skd
    offs_v_base = head_group_id * svh + offs_d[None, :] * svd

    # pointers to m and l
    m_prev = tl.zeros([16], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([16], dtype=tl.float32)
    acc = tl.zeros([16, BLOCK_DMODEL], dtype=tl.float32)

    # number of kv blocks to iterate over
    tmp = 1
    tmp = tmp.to(tl.int64)
    end_n = tmp * (BLOCK_KV // BLOCK_N)

    # need to trim depending on KVLEN
    if kv_len < BLOCK_KV + block_offset:
        remaining_keys = (kv_len - block_offset)
        end_n = (remaining_keys + BLOCK_N - 1) // BLOCK_N

    # Load values
    q_vals = tl.load(Q + offs_q, mask=(offs_m[:, None] < N_CTX_Q) , other=0)

    # rescale sm_scale
    sm_scale *= 1.44269504  # 1/log(2)

    # cast to int64
    offs_k_base = offs_k_base.to(tl.int64)
    offs_v_base = offs_v_base.to(tl.int64)
    svn = svn.to(tl.int64)
    offs_n = offs_n.to(tl.int64)
    kv_len = kv_len.to(tl.int64)

    # workaround for compiler error
    k_idx_vals = tl.zeros([BLOCK_N], dtype=tl.int64)

    # loop over blocks
    for i in range(0, end_n):

        # get around compiler error
        if i > 0:
            offs_n += BLOCK_N

        # load k_idx here using offs_k
        k_idx_vals = tl.load(Kidx + offs_kidx, mask=offs_n < kv_len, other=0).to(tl.int64)

        # compute K/V addresses to load
        offs_k = offs_k_base + k_idx_vals[None, :] * skn
        offs_v = offs_v_base + k_idx_vals[:, None] * svn

        # Load values for K (use kv_len to detect last valid key)
        k_vals = tl.load(K + offs_k, mask=(offs_n[None, :] < kv_len) , other=0)

        # compute qk
        qk = tl.dot(q_vals, k_vals)
        qk *= sm_scale

        # mask here
        qk = tl.where(offs_n[None, :] < kv_len, qk, float("-inf"))

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
        v_vals = tl.load(V + offs_v, mask=(offs_n[:, None] < kv_len), other=0)
        acc += tl.dot(p, v_vals)

        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr

        # update offsets
        offs_kidx += BLOCK_N * skin

    # epilogue
    acc = acc / l_prev[:, None]

    # reset offs_m here
    offs_L = kv_block_id * N_CTX_Q + offs_m
    offs_M = kv_block_id * N_CTX_Q + offs_m

    # store L and M
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)

    # store results to output
    offs_o = kv_block_id * sokv + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=(offs_m[:, None] < N_CTX_Q))

@triton.jit
def _reduce_kernel_qk_gen(
        Out_tmp, Out,
        sotz, sotkv, sotm, sotd, # shape = (Z,NUM_KV_BLOCKS,N_CTX_Q,D)
        soz, soh, som, sod, # shape = (Z,H,N_CTX_Q,D)
        head_start_block, head_kv_len,
        L, M,
        Lout, Mout,
        N_CTX_Q, NUM_KV_HEADS,
        BLOCK_M: tl.constexpr, # will load BLOCK_M queries
        BLOCK_N: tl.constexpr, # will compute self attention by blocks of BLOCK_N keys
        BLOCK_KV: tl.constexpr, # will compute this width of keys
        BLOCK_DMODEL: tl.constexpr
    ):
    start_m = tl.program_id(0).to(tl.int64) # idx of sequence length chunk of size 128 (BLOCK_N)
    head_id = tl.program_id(1).to(tl.int64) # idx of head

    # initialize KV block parameters
    head_start = tl.load(head_start_block + head_id).to(tl.int64) # starting block for this head (used to compute KV indices)
    kv_len = tl.load(head_kv_len + head_id).to(tl.int64) # number of valid KV vectors for this head

    # number of output blocks to loop over
    num_o_tmp_blocks = (kv_len + BLOCK_KV - 1) // BLOCK_KV # number of o_tmp blocks to reduce over

    # get query start indices
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64) # indices of queries we want to process
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64) # we want to process all the dimensions of a given head

    # initialize offsets here
    offs_lm = head_start * N_CTX_Q + offs_m
    offs_o_tmp = head_start * sotkv + offs_m[:, None] * sotm + offs_d[None, :] * sotd
    offs_o = head_id * soh + offs_m[:, None] * som + offs_d[None, :] * sod

    # pointers to m and l - use M and L from causal step as input
    offs_M = head_id * N_CTX_Q + offs_m
    offs_L = head_id * N_CTX_Q + offs_m
    m_prev = tl.load(Mout + offs_M, mask=(offs_m < N_CTX_Q), other=float("-inf")).to(tl.float32)
    l_prev = tl.load(Lout + offs_L, mask=(offs_m < N_CTX_Q), other=0).to(tl.float32)

    # initialize accumulator using output tile
    acc = tl.load(Out + offs_o, mask=(offs_m[:, None] < N_CTX_Q), other=0).to(tl.float32)

    # loop over blocks
    for _ in range(0, num_o_tmp_blocks):

        # load out_tmp block
        o_tmp_vals = tl.load(Out_tmp + offs_o_tmp, mask=offs_m[:, None] < N_CTX_Q, other=0)

        # Load current L / M
        m_curr = tl.load(M + offs_lm, mask=offs_m < N_CTX_Q, other=float("-inf"))
        l_curr = tl.load(L + offs_lm, mask=offs_m < N_CTX_Q, other=0)

        # scale tiles up by denom before adjusting
        acc *= l_prev[:, None]
        o_tmp_vals *= l_curr[:, None]

        # compute largest value from both new block and accum
        m_tmp = tl.maximum(m_curr, m_prev) # compute new m

        # amount to shift by for M1 / M2
        shift1 = tl.math.exp2(m_prev - m_tmp)
        shift2 = tl.math.exp2(m_curr - m_tmp)

        # adjust denominators using largest value from both new block and accum
        l_prev *= shift1 # correct old l
        l_curr *= shift2 # correct old l

        # rescale acc and o_tmp_vals
        acc *= shift1[:, None] # correct old l
        o_tmp_vals *= shift2[:, None] # correct old l

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


@triton.jit
def compact_indices_kernel(
    Mask_ptr, Out_ptr, Num_kv_ptr,
    N_CTX_KV, H,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    # compute global offsets
    idx = tl.arange(0, BLOCK_SIZE)
    offsets = idx + block_id * BLOCK_SIZE
    mask = offsets < N_CTX_KV

    # load mask
    mask_ptr = Mask_ptr + batch_id * H * N_CTX_KV + head_id * N_CTX_KV + offsets
    mask_values = tl.load(mask_ptr, mask=mask, other=0).to(tl.int32)

    # compute cumsum
    cumsum = tl.cumsum(mask_values, axis=0)

    # total valid elements in this block
    total_valid = tl.sum(mask_values)

    # get global offset using atomic add
    global_offset_ptr = Num_kv_ptr + batch_id * H + head_id
    block_start = tl.atomic_add(global_offset_ptr, total_valid)

    # compute write positions
    write_positions = block_start + cumsum - 1

    # store valid indices
    out_ptr = Out_ptr + batch_id * H * N_CTX_KV + head_id * N_CTX_KV
    valid_mask = mask_values == 1
    tl.store(out_ptr + write_positions, offsets, mask=valid_mask)

# simple centroid lookup kernel
@triton.jit
def _fwd_centroid_simple_kernel_qk(
    Q, K, QK,
    NKeys, # shape = (Z,H,N_CENTROIDS)
    CLabels, # shape = (Z,H,N_CTX_KV)
    sm_scale,
    Out,
    ScoreAvg,
    sqz, sqh, sqm, sqd, # shape = (Z,H,N_CTX_Q,D)
    skz, skh, skn, skd, # shape = (Z,H,N_CENTROIDS,D)
    soz, soh, son, som, # shape = (Z,H,N_CENTROIDS,N_CTX_Q)
    ssz, ssh, ssn, ssm, # shape = (Z,H,N_CTX_KV,num_query_blocks)
    sqkz, sqkh, sqkm, sqkn,
    Z, H, N_CTX_Q, N_CTX_KV, N_CENTROIDS,
    BLOCK_M: tl.constexpr, # will load BLOCK_M queries
    BLOCK_N: tl.constexpr, # will compute self attention by blocks of BLOCK_N keys
    BLOCK_DMODEL: tl.constexpr # dimensionality of heads: D
):
    start_m = tl.program_id(0).to(tl.int64) # idx of sequence length chunk of size 128 (BLOCK_N)
    head_id = tl.program_id(1).to(tl.int64) # idx of head

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # indices for QK
    offs_q = head_id * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    offs_k = head_id * skh + offs_n[None, :] * skn + offs_d[:, None] * skd

    # indices for num keys per centroid
    offs_nk = head_id * N_CENTROIDS + offs_n

    # indices for qk res
    offs_qk = head_id * sqkh + offs_n[None, :] * sqkn + offs_m[:, None] * sqkm

    # need to trim depending on N_CENTROIDS
    tmp = 1
    tmp = tmp.to(tl.int64)
    end_n = (N_CENTROIDS + BLOCK_N - tmp) // BLOCK_N

    # Load values
    q_vals = tl.load(Q + offs_q, mask=(offs_m[:, None] < N_CTX_Q) , other=0)

    # store estimate denominator
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)

    sm_scale *= 1.44269504  # 1/log(2)

    # loop over blocks
    for _ in range(0, end_n):

        # Load values for K (use kv_len to detect last valid key)
        k_vals = tl.load(K + offs_k, mask=(offs_n[None, :] < N_CENTROIDS) , other=0)

        # load number of keys per centroids here
        nkeys = tl.load(NKeys + offs_nk, mask=offs_n < N_CENTROIDS, other=0)

        active = (offs_n < N_CENTROIDS) & (nkeys > 0)

        # compute qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16)
        qk += tl.dot(q_vals, k_vals)
        qk *= sm_scale

        # mask out beyond seqlen
        #qk = tl.where(offs_n[None, :] < N_CENTROIDS, qk, float("-inf"))
        qk = tl.where(active[None, :], qk, float("-inf"))

        # save qk
        tl.store(QK + offs_qk, qk, mask=(offs_n[None, :] < N_CENTROIDS) & (offs_m[:, None] < N_CTX_Q))

        # normalization for numerical stability
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        qk = qk - m_curr[:, None]
        p = tl.math.exp2(qk)

        # compute exp_qk * num_c, then sum
        p_tmp = p * nkeys[None, :] # masking is implictly handled here
        l_tmp = tl.sum(p_tmp, 1)
        alpha = tl.math.exp2(m_prev - m_curr)
        denom = denom * alpha + l_tmp

        # update m_prev
        m_prev = m_curr

        # update offsets
        offs_n += BLOCK_N
        offs_nk += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_qk += BLOCK_N

    # need to trim depending on N_CENTROIDS
    tmp = 1
    tmp = tmp.to(tl.int64)
    end_n = (N_CENTROIDS + BLOCK_N - tmp) // BLOCK_N

    # reset n offset
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)

    # base k offset
    offs_k = head_id * skh + offs_n[None, :] * skn + offs_d[:, None] * skd

    # offset for max output score, shape = (Z,H,N_CTX_KV,num_query_blocks)
    offs_s = head_id * ssh + offs_n * ssn + start_m * ssm

    # loop over blocks
    for _ in range(0, end_n):
        # Load values for K (use kv_len to detect last valid key)
        k_vals = tl.load(K + offs_k, mask=(offs_n[None, :] < N_CENTROIDS) , other=0)

        # compute qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16)
        qk += tl.dot(q_vals, k_vals)
        qk *= sm_scale

        # rescale by max
        qk = qk - m_prev[:, None]
        qk = tl.math.exp2(qk)
        qk = qk / denom[:, None]

        # mask here
        m1 = offs_n[None, :] < N_CENTROIDS
        m2 = offs_m[:, None] < N_CTX_Q
        mask = m1 & m2

        # mask qk
        qk = tl.where(mask, qk, 0)

        # compute mean
        qk_avg = tl.sum(qk, 0)

        # write out min/max inverse alpha across query tokens
        tl.store(ScoreAvg + offs_s, qk_avg, mask=offs_n < N_CENTROIDS)

        # update offsets
        offs_n += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_s += BLOCK_N * ssn

@triton.jit
def _fwd_centroid_kernel_attn(
    V, QK,
    NKeys, # shape = (Z,H,N_CENTROIDS)
    sm_scale,
    Out,
    Mask,
    svz, svh, svn, svd, # shape = (Z,H,N_CENTROIDS,D)
    sqkz, sqkh, sqkm, sqkn,
    soz, soh, sokv, som, sod, # shape = (Z,H,N_CTX_Q,D)
    smz, smh, smn, # shape = (Z,H,N_CTX_KV)
    Z, H, N_CTX_Q, N_CENTROIDS,
    L, M,
    num_kv_blocks,
    BLOCK_M: tl.constexpr, # will load BLOCK_M queries
    BLOCK_N: tl.constexpr, # will compute self attention by blocks of BLOCK_N keys
    BLOCK_DMODEL: tl.constexpr, # dimensionality of heads: D
    BLOCK_KV: tl.constexpr, # will load BLOCK_M queries
):
    start_m = tl.program_id(0).to(tl.int64) # idx of sequence length chunk of size 128 (BLOCK_N)
    head_id = tl.program_id(1).to(tl.int64) # idx of head
    kv_split_id = tl.program_id(2).to(tl.int64) # idx of kv block

    assert (start_m == 0) # not supporting batched inference yet

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64)
    block_offset = kv_split_id * BLOCK_KV
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64) + block_offset
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # indices for QKV
    offs_qk = head_id * sqkh + offs_n[None, :] * sqkn + offs_m[:, None] * sqkm
    offs_v = head_id * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
    offs_msk = head_id * smh + offs_n * smn

    # indices for num keys per centroid
    offs_nk = head_id * N_CENTROIDS + offs_n

    # need to trim depending on N_CENTROIDS
    tmp = 1
    tmp = tmp.to(tl.int64)
    # end_n = (N_CENTROIDS + BLOCK_N - tmp) // BLOCK_N
    end_n = tmp * (BLOCK_KV // BLOCK_N)

    # need to trim depending on KVLEN
    if N_CENTROIDS < BLOCK_KV + block_offset:
        remaining_keys = (N_CENTROIDS - block_offset)
        end_n = (remaining_keys + BLOCK_N - 1) // BLOCK_N

    # pointers to m and l
    #m_prev = float("-inf")
    #l_prev = 0.0
    m_prev = tl.zeros([16], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([16], dtype=tl.float32)
    acc = tl.zeros([16, BLOCK_DMODEL], dtype=tl.float32)

    # rescale sm_scale
    sm_scale *= 1.44269504  # 1/log(2)

    # loop over blocks
    for _ in range(0, end_n):

        # load number of keys per centroids here
        nkeys = tl.load(NKeys + offs_nk, mask=offs_n < N_CENTROIDS, other=0)
        mask = tl.load(Mask + offs_msk, mask=offs_n < N_CENTROIDS, other=0)

        active = (mask == 1) & (nkeys > 0)

        if tl.max(active) == 1:
            qk = tl.load(QK + offs_qk, mask=(offs_n[None, :] < N_CENTROIDS) & (offs_m[:, None] < N_CTX_Q), other=float('-inf'))
            qk = tl.where(active[None, :], qk, float('-inf'))

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
            #p_tmp = p_tmp.to(QK.dtype.element_ty)
            v_vals = tl.load(V + offs_v, mask=(offs_n[:, None] < N_CENTROIDS) , other=0).to(tl.float32)

            # # scale values by nkeys
            # v_vals = (v_vals * nkeys[:, None])#.to(V.dtype.element_ty)
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
    offs_L = head_id * num_kv_blocks * N_CTX_Q + kv_split_id * N_CTX_Q + offs_m # split along kv dim
    offs_M = head_id * num_kv_blocks * N_CTX_Q + kv_split_id * N_CTX_Q + offs_m # split along kv dim
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)

    # store results to output
    offs_o = head_id * soh + kv_split_id * sokv + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=(offs_m[:, None] < N_CTX_Q))

@triton.jit
def _reduce_kernel_attn(
        Out_tmp, Out,
        sotz, soth, sotkv, sotm, sotd,
        soz, soh, som, sod, # shape = (Z,H,N_CTX_Q,D)
        L, M,
        Lout, Mout,
        N_CTX_Q, N_CTX_KV,
        BLOCK_M: tl.constexpr,
        BLOCK_KV: tl.constexpr, # will compute this width of keys
        BLOCK_DMODEL: tl.constexpr
    ):

    start_m = tl.program_id(0).to(tl.int64) # idx of sequence length chunk of size 128 (BLOCK_N)
    head_id = tl.program_id(1).to(tl.int64) # idx of head

    # number of output blocks to loop over
    num_o_tmp_blocks = (N_CTX_KV.to(tl.int64) + BLOCK_KV - 1) // BLOCK_KV # number of o_tmp blocks to reduce over
    num_o_tmp_blocks = num_o_tmp_blocks.to(tl.int64)

    # NOTE num_o_tmp_blocks == num_kv_blocks here

    # get query start indices
    offs_m = start_m * BLOCK_M + tl.arange(0, 16).to(tl.int64) # indices of queries we want to process
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64) # we want to process all the dimensions of a given head

    # initialize offsets here
    # offs_lm = head_id * num_o_tmp_blocks
    offs_lm = head_id * num_o_tmp_blocks * N_CTX_Q + 0 * N_CTX_Q + offs_m
    offs_o_tmp = head_id * soth + offs_m[:, None] * sotm + offs_d[None, :] * sotd # add on sotkv to iterate over all kv blocks
    offs_o = head_id * soh + offs_m[:, None] * som + offs_d[None, :] * sod

    offs_lm = offs_lm.to(tl.int64)
    offs_o_tmp = offs_o_tmp.to(tl.int64)
    offs_o = offs_o.to(tl.int64)

    # pointers to m and l - use M and L from causal step as input
    offs_M = head_id * N_CTX_Q + offs_m
    offs_L = head_id * N_CTX_Q + offs_m
    # offs_L = head_id * num_o_tmp_blocks * N_CTX_Q + 0 * N_CTX_Q + offs_m # split along kv dim
    # offs_M = head_id * num_o_tmp_blocks * N_CTX_Q + 0 * N_CTX_Q + offs_m # split along kv dim
    m_prev = tl.load(Mout + offs_M, mask=offs_m < N_CTX_Q, other=-float("inf")).to(tl.float32) # do a single load instead of vector load
    l_prev = tl.load(Lout + offs_L, mask=offs_m < N_CTX_Q, other=0).to(tl.float32)

    # initialize accumulator using output tile
    acc = tl.load(Out + offs_o, mask=(offs_m[:, None] < N_CTX_Q), other=0).to(tl.float32)

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
        m_tmp = tl.maximum(m_curr, m_prev) # compute new m

        # amount to shift by for M1 / M2
        shift1 = tl.math.exp2(m_prev - m_tmp)
        shift2 = tl.math.exp2(m_curr - m_tmp)

        # adjust denominators using largest value from both new block and accum
        l_prev *= shift1 # correct old l
        l_curr *= shift2 # correct old l

        # rescale acc and o_tmp_vals
        acc *= shift1[:, None] # correct old l
        o_tmp_vals *= shift2[:, None] # correct old l

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

    # store L and M updates as well.
    tl.store(Lout + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(Mout + offs_M, m_prev, mask=offs_m < N_CTX_Q)

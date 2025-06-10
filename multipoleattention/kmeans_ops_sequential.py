import torch
from typing import Literal, Optional

# fast distance function (faster than cdist)
@torch.compile(mode="reduce-overhead")
def _sq_euclid(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x2 = x.pow(2).sum(-1, keepdim=True)
    c2 = c.pow(2).sum(-1).unsqueeze(1)
    prod = torch.matmul(x, c.transpose(-1, -2))
    return x2 + c2 - 2 * prod

@torch.compile
def block_kmeans(
    data: torch.Tensor,
    num_clusters: int,
    lloyd_iters: int = 10,
    chunk_size: int = 16384,
):

    H, N, D = data.shape
    device  = data.device
    dtype   = data.dtype

    # clamp K so we never request more clusters than points
    K = min(num_clusters, N)

    # random‐seed K centroids from the block (separate indices per head)
    idx0 = torch.stack([torch.randperm(N, device=device)[:K] for _ in range(H)], dim=0)
    centroids = torch.gather(data, 1, idx0.unsqueeze(-1).expand(-1, -1, D))

    # initialize count / sum tensors
    counts = torch.zeros((H, K), dtype=torch.int32,   device=device)
    sum_K  = torch.zeros((H, K, D), dtype=torch.float32, device=device)

    # tensor for scatter operation
    ones_chunk = torch.ones((H, chunk_size), dtype=torch.int32, device=device)

    # lloyd iters
    for i in range(lloyd_iters):

        # reset accumulators
        counts.zero_()
        sum_K.zero_()

        # chunked assign & accumulate
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            b   = end - start
            x_chunk = data[:, start:end]

            # compute distances to centroids
            d2 = _sq_euclid(x_chunk.half(), centroids.half())
            assign  = d2.argmin(dim=-1)

            # accumulate counts
            counts.scatter_add_(1, assign, ones_chunk[:, :b])

            # accumulate sums
            sum_K.scatter_add_(
                1,
                assign.unsqueeze(-1).expand(-1, -1, D),
                x_chunk.float()
            )

        # handling empty clusters (not on last iteration)
        empty = counts == 0
        if empty.any() and i < lloyd_iters - 1:
            # for each head h, pick a random point for each empty cluster
            for h in range(H):
                em = torch.nonzero(empty[h], as_tuple=True)[0]
                if em.numel() == 0:
                    continue

                # sample one random index per empty cluster
                idx_new = torch.randint(0, N, (em.numel(),), device=device)
                centroids[h, em] = data[h, idx_new]
                counts[h, em] = 1.0
                sum_K[h, em] = centroids[h, em].float()

        # update centroids for non‐empty clusters
        denom = counts.unsqueeze(-1)
        centroids = torch.where(
            denom > 0,
            (sum_K / denom.clamp_min(1)).to(dtype),
            centroids
        )

    # final labels
    labels = torch.empty((H, N), dtype=torch.long, device=device)
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        x   = data[:, start:end]
        labels[:, start:end] = _sq_euclid(x.half(), centroids.half()).argmin(-1)

    return centroids, counts.to(dtype), labels

@torch.compile
def sequential_kmeans_append(
    data_new:      torch.Tensor,
    centroids_old: torch.Tensor,
    counts_old:    torch.Tensor,
    refine_data:    torch.Tensor,
    k_extra: int,
    lloyd_iters: int = 3,
    chunk_size: int = 16384,
):
    # unpack shapes & allocate output
    H, M, D   = data_new.shape
    _, N_refine, _ = refine_data.shape
    K0        = centroids_old.size(1)
    k_extra   = min(k_extra, M)
    K_tot     = K0 + k_extra
    device    = data_new.device
    dtype     = data_new.dtype

    centroids = torch.empty((H, K_tot, D), dtype=dtype,  device=device)
    counts    = torch.zeros((H, K_tot),    dtype=torch.float32, device=device)
    centroids[:, :K0] = centroids_old
    counts[:,   :K0]  = counts_old

    # random‑seed k_extra per head
    idx_extra = torch.stack([torch.randperm(M, device=device)[:k_extra] for _ in range(H)],dim=0)
    centroids[:, K0:] = torch.gather(data_new.float(), 1, idx_extra.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones((H, M), dtype=torch.bool, device=device)
    mask.scatter_(1, idx_extra, False)
    residual = data_new[mask].view(H, M - k_extra, D)
    B = residual.size(1)

    # convenience tensor
    ones_B = torch.ones((H, B), dtype=torch.float32, device=device)

    # one‑shot mini‑batch update
    d2 = torch.cdist(residual.float(), centroids.float(), p=2)
    assign = d2.argmin(-1)

    batch_cnt = torch.zeros_like(counts)
    batch_cnt.scatter_add_(1, assign, ones_B)

    batch_sum = torch.zeros_like(centroids, dtype=torch.float32)
    batch_sum.scatter_add_(1,
                           assign.unsqueeze(-1).expand(-1, -1, D),
                           residual.float())

    counts += batch_cnt
    centroids = (centroids * (counts - batch_cnt).unsqueeze(-1) +
                 batch_sum) / counts.clamp_min(1).unsqueeze(-1)

    # tensor for scatter operation
    ones_chunk = torch.ones((H, chunk_size), dtype=counts.dtype, device=device)

    # Lloyd refinement passes
    for i in range(lloyd_iters):
        centroids_half = centroids.half()   # keep one FP16 copy per iter
        cnt_ll = torch.zeros_like(counts)
        sum_ll = torch.zeros_like(centroids).float()

        for start in range(0, N_refine, chunk_size):
            end = min(N_refine, start + chunk_size)
            b   = end - start
            x_chunk = refine_data[:, start:end]

            # compute distances to centroids
            d2 = _sq_euclid(x_chunk.half(), centroids_half)
            assign  = d2.argmin(dim=-1)

            # accumulate counts
            #cnt_ll.scatter_add_(1, assign, ones_chunk[:, :b])
            ones_b = torch.ones((H, b), dtype=torch.float32, device=device)
            cnt_ll.scatter_add_(1, assign, ones_b)

            # accumulate sums
            sum_ll.scatter_add_(
                1,
                assign.unsqueeze(-1).expand(-1, -1, D),
                x_chunk.float()
            )

        # handling empty clusters (not on last iteration)
        empty = cnt_ll == 0
        if empty.any() and i < lloyd_iters - 1:
            # for each head h, pick a random point for each empty cluster
            for h in range(H):
                em = torch.nonzero(empty[h], as_tuple=True)[0]
                if em.numel() == 0:
                    continue
                idx_new = torch.randint(0, N_refine, (em.numel(),), device=device)
                picked = refine_data[h, idx_new].float()
                sum_ll[h, em] = picked
                cnt_ll[h, em] = 1.0
                centroids[h, em] = picked.to(centroids.dtype)

        # update centroids for non‐empty clusters
        denom = cnt_ll.unsqueeze(-1)
        centroids = torch.where(
            denom > 0,
            (sum_ll / denom.clamp_min(1)).to(dtype),
            centroids
        )

    # get the final labels
    labels = torch.empty((H, N_refine), dtype=torch.long, device=device)
    for s in range(0, N_refine, chunk_size):
        e = min(N_refine, s + chunk_size)
        labels[:, s:e] = _sq_euclid(refine_data[:, s:e].half(), centroids.half()).argmin(-1)

    return centroids.to(data_new.dtype), counts, labels

from concurrent.futures import ThreadPoolExecutor
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize
import torch.multiprocessing as mp
import math
from torch import cdist
from multipoleattention.kmeans_ops_sequential import sequential_kmeans_append, block_kmeans

def run_clustering_online_onelevel(key_states, K, rank=0):
    device = torch.device(f"cuda:{rank}")
    _, H, N_total, D = key_states.shape

    # determine final 8–24K window
    _, H, N_total, D = key_states.shape
    BLOCK = 8_192
    MIN_FINAL  = 4_096
    MAX_FINAL  = 12_288

    # determine number of blocks
    n_blocks = N_total // BLOCK
    rem      = N_total - n_blocks * BLOCK

    if rem and rem < MIN_FINAL:
        n_blocks  = max(0, n_blocks - 1)
        rem       = N_total - n_blocks * BLOCK

    block_sizes = [BLOCK] * n_blocks + ([rem] if rem else [])
    assert 0 < block_sizes[-1] <= MAX_FINAL

    # decide per‑block cluster counts
    cluster_frac = K / N_total
    k_per_block  = [math.floor(cluster_frac * b) for b in block_sizes]
    assert sum(k_per_block) == K
    assert all(k > 0 for k in k_per_block)

    # targets
    cluster_centers, cluster_labels, cluster_counts = [], [], []
    next_cluster_base = 0

    # iterate over blocks
    for bsz, k_block in zip(block_sizes, k_per_block):
        start = len(cluster_labels) * BLOCK
        end   = start + bsz

        block_data = key_states[0, :, start:end]
        C, cnt, lbl = block_kmeans(block_data, k_block)

        cluster_centers.append(C)
        cluster_counts.append(cnt)
        cluster_labels.append(lbl + next_cluster_base)
        next_cluster_base += k_block

    # stack outputs
    centroids_tensor  = torch.cat(cluster_centers, dim=1).unsqueeze(0)
    centroids_labels  = torch.cat(cluster_labels, dim=1).unsqueeze(0).to(torch.int64)
    num_keys_tensor   = torch.cat(cluster_counts, dim=1).unsqueeze(0).to(torch.int64)

    return centroids_tensor, centroids_labels, num_keys_tensor

def run_clustering_online(keys, num_clusters, num_levels, rank=None):
    assert (rank is not None)

    # hierarchical centroids with multi-level implementation
    centroid_tensors_all_levels = [0] * num_levels
    centroid_labels_all_levels = [0] * num_levels
    num_keys_list = [0] * num_levels
    centroid_tensors_all_levels_orig = [0] * num_levels

    # iterate in reverse and run k-means hierarchically
    centroid_tensor = keys
    for level in range(num_levels - 1, -1, -1):
        centroid_tensor, centroid_label, num_keys = run_clustering_online_onelevel(centroid_tensor,
                                                                num_clusters[level],
                                                                rank=rank)
        centroid_tensors_all_levels[level] = centroid_tensor
        centroid_labels_all_levels[level] = centroid_label
        num_keys_list[level] = num_keys
        centroid_tensors_all_levels_orig[level] = centroid_label

    # update centroids_labels_dict_l1
    for j in range(num_levels - 1, 0, -1):
        # update centroid labels
        label_dct_higher = centroid_labels_all_levels[j - 1]
        label_dct_lower = centroid_labels_all_levels[j]
        gathered_tensor = torch.gather(label_dct_higher, -1, label_dct_lower)
        centroid_labels_all_levels[j - 1] = gathered_tensor

    return centroid_tensors_all_levels, centroid_labels_all_levels, num_keys_list, centroid_tensors_all_levels_orig

def run_clustering_online_onelevel_update(
    key_states,
    prev_clusters,
    prev_clusters_labels,
    prev_num_keys,
    num_new_clusters,
    num_new_tokens,
    rank: int = 0,
):
    device = torch.device(f"cuda:{rank}")

    # determine final 8–24K window
    _, H, N_total, D = key_states.shape
    _, _H, K_old, _D  = prev_clusters.shape
    BLOCK_SIZE = 8_192
    MIN_FINAL  = 4_096
    MAX_FINAL  = 12_288

    # total sizes after the append
    K_total = K_old + num_new_clusters
    N_old   = prev_clusters_labels.shape[2]
    N_total_check = N_old + num_new_tokens

    # assume number of tokens each step is divisible by num clusters
    cluster_percentage = num_new_clusters / num_new_tokens
    assert((num_new_tokens % num_new_clusters) == 0)

    # how many full 16K blocks
    n_blocks = max(0, N_total // BLOCK_SIZE)
    rem      = N_total - n_blocks * BLOCK_SIZE

    # if the leftover is too small, peel off one more 16K so final ≥ 8K
    if rem < MIN_FINAL and n_blocks:
        n_blocks -= 1
        rem       = N_total - n_blocks * BLOCK_SIZE
    assert 0 < rem <= MAX_FINAL, "final window size out of bounds"

    # allocate merged outputs
    centroids_tensor = torch.empty((H, K_total, D), dtype=torch.bfloat16, device=device)
    centroids_labels = torch.empty((H, N_total), dtype=torch.int64, device=device)
    num_keys_tensor = torch.empty((H, K_total), dtype=torch.int64, device=device)

    # get offsets to copy
    copy_offset = n_blocks * BLOCK_SIZE
    centroid_copy_offset = int(copy_offset * cluster_percentage)

    centroids_tensor[:,:centroid_copy_offset] = prev_clusters[:,:,:centroid_copy_offset]
    centroids_labels[:,:copy_offset] = prev_clusters_labels[:,:,:copy_offset]
    num_keys_tensor[:,:centroid_copy_offset] = prev_num_keys[:,:,:centroid_copy_offset]

    # special 24K case: block_kmeans and sequential_kmeans_append
    if rem == MAX_FINAL:
        # compute offset of window
        start_offset = n_blocks * BLOCK_SIZE
        end_offset = (n_blocks + 1) * BLOCK_SIZE
        window_data  = key_states[0, :, start_offset:end_offset].to(device)
        num_clusters = int(BLOCK_SIZE * cluster_percentage)

        centroid_start_offset = int(start_offset * cluster_percentage)
        centroid_end_offset = int(end_offset * cluster_percentage)

        # refine first 16K of the window
        cent_interim, num_keys_interim, labels_interim = block_kmeans(data_new, num_clusters)

        # copy data over
        centroids_tensor[:,centroid_start_offset:centroid_end_offset] = cent_interim
        centroids_labels[:,start_offset:end_offset] = labels_interim + centroid_start_offset
        num_keys_tensor[:,centroid_start_offset:centroid_end_offset] = num_keys_interim.to(num_keys_tensor.dtype)

        # increment before hitting last window
        n_blocks += 1

    # compute offset of window
    start_offset = n_blocks * BLOCK_SIZE

    # new data - only the ones to append
    # window data - full window to cluster over
    new_data  = key_states[0, :, -num_new_tokens:].to(device)
    window_data  = key_states[0, :, start_offset:].to(device)

    # get centroid offset
    centroid_start_offset = int(start_offset * cluster_percentage)

    # bring in old centroids & counts
    centroids_old = prev_clusters[0, :, centroid_start_offset:].to(device)
    counts_old    = prev_num_keys[0, :, centroid_start_offset:].to(device)

    # single-shot append over the entire final window
    centroids_upd, num_keys_upd, labels_upd = sequential_kmeans_append(
        data_new      = new_data,
        centroids_old = centroids_old,
        counts_old    = counts_old,
        refine_data   = window_data,
        k_extra       = num_new_clusters,
    )

    # copy data over
    centroids_tensor[:,centroid_start_offset:] = centroids_upd
    centroids_labels[:,start_offset:] = labels_upd + centroid_start_offset
    num_keys_tensor[:,centroid_start_offset:] = num_keys_upd

    centroids_tensor = centroids_tensor.unsqueeze(0).to(torch.bfloat16)
    centroids_labels = centroids_labels.unsqueeze(0)
    num_keys_tensor  = num_keys_tensor.unsqueeze(0)

    return centroids_tensor, centroids_labels, num_keys_tensor

def run_clustering_online_update(
        keys, prev_clusters_lst, prev_cluster_labels_list,
        prev_num_keys_list, num_new_clusters_list,
        num_new_tokens_list, num_levels, rank=None
    ):
    assert (rank is not None)

    # Hierarchical centroids with multi-level implementation
    centroid_tensors_all_levels = [0] * num_levels
    centroid_labels_all_levels = [0] * num_levels
    num_keys_list = [0] * num_levels
    centroid_tensors_all_levels_orig = [0] * num_levels

    # do this backwards as we calculate from lowest level to highest.
    centroid_tensor = keys
    for level in range(num_levels - 1, -1, -1):
        centroid_tensor, centroid_label, num_keys = run_clustering_online_onelevel_update(
            centroid_tensor,
            prev_clusters_lst[level],
            prev_cluster_labels_list[level],
            prev_num_keys_list[level],
            num_new_clusters_list[level],
            num_new_tokens_list[level],
            rank=rank
        )
        centroid_tensors_all_levels[level] = centroid_tensor
        centroid_labels_all_levels[level] = centroid_label
        num_keys_list[level] = num_keys
        centroid_tensors_all_levels_orig[level] = centroid_label

    # update centroids_labels_dict_l1
    for j in range(num_levels - 1, 0, -1):
        # update centroid labels
        label_dct_higher = centroid_labels_all_levels[j - 1]
        label_dct_lower = centroid_labels_all_levels[j]
        gathered_tensor = torch.gather(label_dct_higher, -1, label_dct_lower)
        centroid_labels_all_levels[j - 1] = gathered_tensor

    return centroid_tensors_all_levels, centroid_labels_all_levels, num_keys_list, centroid_tensors_all_levels_orig

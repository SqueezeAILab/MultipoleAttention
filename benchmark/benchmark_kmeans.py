import time
import torch
import triton
import triton.language as tl
from multipoleattention.kmeans_ops_sequential import sequential_kmeans_append, block_kmeans
torch.set_float32_matmul_precision('high')

# configuration
DTYPE, DEVICE = torch.float32, "cuda:0"
H, N0, D, K0 = 8, 8192, 128, 512 # number of KV heads, context length, head dim, number of centroids
M,  K_DELTA = 128, 8 # number of new tokens, number of new centroids (one per 16 tokens)
REFINE_ITERS = 3 # lloyd refinement passes
torch.manual_seed(0)

# random data
data0 = torch.randn(H, N0, D, dtype=DTYPE, device=DEVICE)
data_new = torch.randn(H, M,  D, dtype=DTYPE, device=DEVICE)
all_data = torch.cat([data0, data_new], dim=1)

# run initial blocked kmeans
print("Initial K‑Means on 8K (512 clusters)\n")
centroids_A, counts_A, labels_A = block_kmeans(data0, K0)

# fast online cluster update
print("Fast Online Cluster Update\n")
fn = lambda: sequential_kmeans_append(data_new, centroids_A, counts_A, all_data, K_DELTA, lloyd_iters=REFINE_ITERS)
T_C = triton.testing.do_bench(fn, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])[1]
print(f"{T_C:8.1f} ms")

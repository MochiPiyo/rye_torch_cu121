import torch
import time

size = 1000
sparsity = 0.1

# 密行列をランダムに生成し、CUDAに送る
dense1 = torch.randn(size, size).cuda()
dense2 = torch.randn(size, size).cuda()

# 密行列の行列乗算の時間とメモリ消費量を計測
start_time = time.time()
dense_result = torch.matmul(dense1, dense2)
end_time = time.time()
dense_memory = torch.cuda.memory_allocated()  # メモリ消費量を取得
print(f"Dense matrix multiplication took {end_time - start_time} seconds, Memory: {dense_memory} bytes")

# スパーシティに基づいてランダムなマスクを生成し、CUDAに送る
mask1 = torch.lt(torch.rand(size, size), sparsity)
mask2 = torch.lt(torch.rand(size, size), sparsity)

# 密行列にマスクを適用して疎行列を生成し、CUDAに送る
sparse1 = torch.randn(size, size).masked_fill_(mask1, 0).to_sparse().cuda()
sparse2 = torch.randn(size, size).masked_fill_(mask2, 0).to_sparse().cuda()

# 疎行列の行列乗算の時間とメモリ消費量を計測
start_time = time.time()
sparse_result = torch.sparse.mm(sparse1, sparse2)
end_time = time.time()
sparse_memory = torch.cuda.memory_allocated()  # メモリ消費量を取得
print(f"Sparse matrix with sparsity '{sparsity}' multiplication took {end_time - start_time} seconds, Memory: {sparse_memory} bytes")

# 疎行列をランダムに生成し、CUDAに送る
sparse1 = torch.rand(size, size).cuda().to_sparse()
sparse2 = torch.rand(size, size).cuda().to_sparse()

# 疎行列の行列乗算の時間とメモリ消費量を計測
start_time = time.time()
sparse_result = torch.sparse.mm(sparse1, sparse2)
end_time = time.time()
sparse_memory = torch.cuda.memory_allocated()  # メモリ消費量を取得
print(f"Sparse matrix with dense element multiplication took {end_time - start_time} seconds, Memory: {sparse_memory} bytes")

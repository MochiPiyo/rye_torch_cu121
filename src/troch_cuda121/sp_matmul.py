import torch
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

size = 2000
sparsity = 0.1

"""
明らかに密行列の方が速い
size = 2000
sparsity = 0.1

Dense matrix multiplication took 0.2627570629119873 seconds
Sparse matrix with sparsity '0.1' multiplication took 9.670902013778687 seconds
Sparse matrix with dense element multiplication took 10.970580577850342 seconds
"""

# 密行列をランダムに生成
dense1 = torch.randn(size, size)
dense2 = torch.randn(size, size)

# 密行列の行列乗算の時間を計測
start_time = time.time()
dense_result = torch.matmul(dense1, dense2)
end_time = time.time()
print(f"Dense matrix multiplication took {end_time - start_time} seconds")

# スパーシティに基づいてランダムなマスクを生成
# torch.ltはless than。randは0..1の一様分布なので0..1のsparsity以下を選ぶことでsparsityで指定した割合だけ残る
mask1 = torch.lt(torch.rand(size, size), sparsity)
mask2 = torch.lt(torch.rand(size, size), sparsity)

# 密行列にマスクを適用して疎行列を生成
sparse1 = dense1.masked_fill_(mask1, 0).to_sparse()
sparse2 = dense2.masked_fill_(mask2, 0).to_sparse()

# 疎行列の行列乗算の時間を計測
start_time = time.time()
sparse_result = torch.sparse.mm(sparse1, sparse2)
end_time = time.time()
print(f"Sparse matrix with sparsity '{sparsity}' multiplication took {end_time - start_time} seconds")



# 疎行列をランダムに生成
sparse1 = torch.rand(size, size).to_sparse()
sparse2 = torch.rand(size, size).to_sparse()

# 疎行列の行列乗算の時間を計測
start_time = time.time()
sparse_result = torch.sparse.mm(sparse1, sparse2)
end_time = time.time()
print(f"Sparse matrix with dense element multiplication took {end_time - start_time} seconds")



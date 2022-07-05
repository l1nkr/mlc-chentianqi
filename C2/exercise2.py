import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import torch
import time
# 其中，A 是输入张量，W 是权重张量，b 是批次索引，k 是输出通道，i 和 j 是图像高度和宽度的索引
# di 和 dj 是权重的索引，q 是 输入通道，strides 是过滤器窗口的步幅。


# something wrong
@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(A: T.Buffer[(1, 1, 8, 8), "int64"],
             B: T.Buffer[(2, 1, 3, 3), "int64"],
             C: T.Buffer[(1, 2, 6, 6), "int64"]):
        T.func_attr({"global_symbol": "conv", "tir.noalias": True})
        for i, j, k, m, n in T.grid(2, 6, 6, 3, 3):
            with T.block("C"):
                vi, vj, vk, vm, vn = T.axis.remap("SSSSS", [i, j, k, m, n])
                C[0, vi, vj, vk] = C[0, vi, vj, vk] + A[0, 0, vj+vm, vk+vn] * B[vi, 0, vm, vn]
                


N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

# torch version

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
start_1 = time.time()
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
end_1 = time.time()
print(end_1 - start_1)
conv_torch = conv_torch.numpy().astype(np.int64)
print(conv_torch)

# python version
def conv2d(data, weight):
    start_2 = time.time()
    out = np.arange(1*2*6*6).reshape(1, 2, 6, 6)
    for i in range(2):
        for j in range(6):
            for k in range(6):
                out[0, i, j, k] = 0
                for m in range(3):
                    for n in range(3):
                        out[0, i, j, k] = out[0, i, j, k] + data[0, 0, j+m, k+n] * weight[i, 0, m, n]
    end_2 = time.time()
    print(end_2 - start_2)
    print(out)
    
conv2d(data, weight)

# TensorIR version
rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
print(conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)

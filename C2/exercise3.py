from os import replace
from turtle import rt
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import IPython
import torch

# example
@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer[(4, 4), "int64"],
            B: T.Buffer[(4, 4), "int64"],
            C: T.Buffer[(4, 4), "int64"]):
        T.func_attr({"global_symbol": "add"})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]

def show_example():
    sch = tvm.tir.Schedule(MyAdd)
    block = sch.get_block("C", func_name="add")
    i, j = sch.get_loops(block)
    i0, i1 = sch.split(i, factors=[2, 2])
    sch.parallel(i0)
    sch.unroll(i1)
    sch.vectorize(j)
    print(IPython.display.Code(sch.mod.script(), language="python"))

# 
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)
                
def test_1():
    # test
    a = np.random.rand(16, 128, 128).astype("float32")
    b = np.random.rand(16, 128, 128).astype("float32")
    c = np.empty((16, 128, 128), dtype="float32")
    lnumpy_mm_relu_v2(a, b, c)

    a_nd = tvm.nd.array(a)
    b_nd = tvm.nd.array(b)
    c_nd = tvm.nd.empty((16, 128, 128), dtype="float32")
    rt_lib = tvm.build(MyBmmRule, target="llvm")
    func_bmm_relu = rt_lib["bmm_relu"]
    func_bmm_relu(a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c, c_nd.numpy(), rtol=1e-5)

    # torch test
    m = torch.nn.ReLU(inplace=True)
    c2 = m(torch.from_numpy(a @ b))
    np.testing.assert_allclose(c, c2.numpy(), rtol=1e-5)

@tvm.script.ir_module
class MyBmmRule:
    @T.prim_func
    def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"],
                 B: T.Buffer[(16, 128, 128), "float32"],
                 C: T.Buffer[(16, 128, 128), "float32"]):
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i, j, k, m in T.grid(16, 128, 128, 128):
            with T.block("Y"):
                vi, vj, vk, vm = T.axis.remap("SSSR", [i, j, k, m])
                with T.init():
                    Y[vi, vj, vk] = T.float32(0)
                Y[vi, vj, vk] = Y[vi, vj, vk] + A[vi, vj, vm] * B[vi, vm, vk]
        for i, j, k in T.grid(16, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                C[vi, vj, vk] = T.max(Y[vi, vj, vk], T.float32(0))
                
# TODO: transformations
sch = tvm.tir.Schedule(MyBmmRule)
# print(IPython.display.Code(sch.mod.script(), language="python"))

# Step 1. Get blocks
Y = sch.get_block("Y", func_name="bmm_relu")
C = sch.get_block("C", func_name="bmm_relu")

# Step 2. Get loops
y_i, y_j, y_k, y_m = sch.get_loops(Y)
c_i, c_j, c_k = sch.get_loops(C)

# Step 3. Organize the loops
y_m0, y_m1 = sch.split(y_j, [32, 8])
sch.reorder(y_m1, y_k, y_m0)
# sch.compute_at(...)
# ...

# Step 4. decompose reduction
# Y_init = sch.decompose_reduction(Y, ...)
# ...

# Step 5. vectorize / parallel / unroll
# sch.vectorize(...)
# sch.parallel(...)
# sch.unroll(...)
# ...

print(IPython.display.Code(sch.mod.script(), language="python"))
# tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")
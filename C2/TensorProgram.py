import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import IPython

@tvm.script.ir_module
class MyModule:
    # @T.prim_func
    # def main(A: T.Buffer[128, "float32"],
    #          B: T.Buffer[128, "float32"],
    #          C: T.Buffer[128, "float32"]):
    #     # extra annotations for the function
    #     T.func_attr({"global_symbol": "main", "tir.noalias": True})
    #     for i in range(128):
    #         with T.block("C"):
    #             # declare a data parallel iterator on spatial domain
    #             vi = T.axis.spatial(128, i)
    #             C[vi] = A[vi] + B[vi]
                
    @T.prim_func
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                # A block is a basic unit of computation in TensorIR
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

dtype = "float32"
a = np.random.rand(128, 128).astype(dtype)
b = np.random.rand(128, 128).astype(dtype)
c = np.maximum(a @ b, 0)
# print(type(MyModule["main"]))
sch = tvm.tir.Schedule(MyModule)
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)
j0, j1 = sch.split(j, factors=[None, 4])
# print(IPython.display.Code(sch.mod.script(), language="python"))
sch.reorder(j0, k, j1)
# print(IPython.display.Code(sch.mod.script(), language="python"))
block_C = sch.get_block("C", "mm_relu")
# sch.reverse_compute_at(block_C, j0)
# print(IPython.display.Code(sch.mod.script(), language="python"))
# sch.decompose_reduction(block_Y, k)
print(IPython.display.Code(sch.mod.script(), language="python"))
rt_lib = tvm.build(MyModule, target="llvm")
a_nd = tvm.nd.array(a)
b_nd = tvm.nd.array(b)
c_nd = tvm.nd.empty((128, 128), dtype="float32")
func_mm_relu = rt_lib["mm_relu"]
# func_mm_relu(a_nd, b_nd, c_nd)
# np.testing.assert_allclose(c, c_nd.numpy(), rtol=1e-5)
rt_lib_after = tvm.build(sch.mod, target="llvm")
func_mm_relu_after = rt_lib_after["mm_relu"]
func_mm_relu_after(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c, c_nd.numpy(), rtol=1e-5)

timer = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % timer(a_nd, b_nd, c_nd).mean)
timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed %g sec" % timer_after(a_nd, b_nd, c_nd).mean)


def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

# dtype = "float32"
# a = np.random.rand(128, 128).astype(dtype)
# b = np.random.rand(128, 128).astype(dtype)
# c = np.maximum(a @ b, 0)
# print(c)

# c_np = np.empty((128, 128), dtype=dtype)
# lnumpy_mm_relu(a, b, c_np)
# print(c_np)
# np.testing.assert_allclose(c, c_np, rtol=1e-5)


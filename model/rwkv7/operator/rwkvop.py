
from einops import rearrange
import os, math, gc, importlib
import torch
########################################################################################################
# CUDA Kernel
########################################################################################################
def RUN_CUDA_RWKV7g():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

def RUN_RWKV7_STATE():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

def RUN_RWKV7_INFCTX():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

def RUN_CUDA_RWKV6():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


def RUN_CUDA_RWKV6_STATE():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


def RUN_CUDA_RWKV5():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')




from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16
    if os.environ["RWKV_TRAIN_TYPE"] == 'state':
        if os.environ["RWKV_FLOAT_MODE"] == 'bf16':
            flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
            load(name="rwkv7_state_clampw", sources=[f'model/rwkv7/cuda/rwkv7_state_clampw.cu', 'model/rwkv7/cuda/rwkv7_state_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

            class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
                @staticmethod
                def forward(ctx,s0,r,w,k,v,a,b):
                    B,T,H,C = r.shape
                    assert T%CHUNK_LEN == 0
                    assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
                    assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
                    assert s0.dtype==torch.float
                    y = torch.empty_like(v)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    sa = torch.empty(B,T,H,C,dtype=torch.float32,device=w.device)
                    torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
                    ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                    return y
                @staticmethod
                def backward(ctx,dy):
                    assert all(i.dtype==torch.bfloat16 for i in [dy])
                    assert all(i.is_contiguous() for i in [dy])
                    r,w,k,v,a,b,s,sa = ctx.saved_tensors
                    B,T,H,C = r.shape
                    dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                    ds0 = torch.empty(B,H,C,C,dtype=torch.float32,device=r.device)
                    torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
                    return ds0,dr,dw,dk,dv,da,db
            def RUN_RWKV7_STATE(r,k,v,w,a,b,s0):
                B,T,HC = r.shape
                C = HEAD_SIZE
                H = HC//C
                s0 = s0.float().repeat(B, 1, 1, 1)
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC), None

        elif os.environ["RWKV_FLOAT_MODE"] == 'fp32':
            flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', "-D_FP32_", f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
            load(name="rwkv7_state_clampw", sources=[f'model/rwkv7/cuda/rwkv7_state_clampw.cu', 'model/rwkv7/cuda/rwkv7_state_clampw.cpp'], is_python_module=False, verbose=True, extra_cflags=["-D_FP32_"], extra_cuda_cflags=flags)

            class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
                @staticmethod
                def forward(ctx,s0,r,w,k,v,a,b):
                    B,T,H,C = r.shape 
                    assert T%CHUNK_LEN == 0
                    assert all(i.dtype==torch.float32 for i in [s0,r,w,k,v,a,b])
                    assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
                    y = torch.empty_like(v)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                    torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
                    ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                    return y
                @staticmethod
                def backward(ctx,dy):
                    assert all(i.dtype==torch.float32 for i in [dy])
                    assert all(i.is_contiguous() for i in [dy])
                    r,w,k,v,a,b,s,sa = ctx.saved_tensors
                    B,T,H,C = r.shape
                    dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                    ds0 = torch.empty(B,H,C,C,dtype=torch.float32,device=r.device)
                    torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
                    return ds0,dr,dw,dk,dv,da,db
            def RUN_RWKV7_STATE(r,k,v,w,a,b,s0):
                B,T,HC = r.shape
                C = HEAD_SIZE
                H = HC//C
                s0 = s0.repeat(B, 1, 1, 1)
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC), None

        else:
            raise NotImplementedError("Unsupported precision for RWKV7 fine-tuning")

    else:
        if os.environ["RWKV_FLOAT_MODE"] == 'bf16':
            flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
            load(name="rwkv7_clampw", sources=[f'model/rwkv7/cuda/rwkv7_clampw.cu', 'model/rwkv7/cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
            class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
                @staticmethod
                def forward(ctx,r,w,k,v,a,b):
                    B,T,H,C = r.shape 
                    assert T%CHUNK_LEN == 0
                    assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
                    assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
                    y = torch.empty_like(v)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                    torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
                    ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                    return y
                @staticmethod
                def backward(ctx,dy):
                    assert all(i.dtype==torch.bfloat16 for i in [dy])
                    assert all(i.is_contiguous() for i in [dy])
                    r,w,k,v,a,b,s,sa = ctx.saved_tensors
                    dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                    torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
                    return dr,dw,dk,dv,da,db

            def RUN_CUDA_RWKV7g(r,w,k,v,a,b):
                B,T,HC = r.shape
                r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
                return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HC)

        elif os.environ["RWKV_FLOAT_MODE"] == 'fp32':
            flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', "-D_FP32_", f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
            load(name="rwkv7_clampw", sources=[f'model/rwkv7/cuda/rwkv7_clampw.cu', 'model/rwkv7/cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=True, extra_cflags=["-D_FP32_"], extra_cuda_cflags=flags)
            class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
                @staticmethod
                def forward(ctx,r,w,k,v,a,b):
                    B,T,H,C = r.shape 
                    assert T%CHUNK_LEN == 0
                    assert all(i.dtype==torch.float32 for i in [r,w,k,v,a,b])
                    assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
                    y = torch.empty_like(v)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                    torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
                    ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                    return y
                @staticmethod
                def backward(ctx,dy):
                    assert all(i.dtype==torch.float32 for i in [dy])
                    assert all(i.is_contiguous() for i in [dy])
                    r,w,k,v,a,b,s,sa = ctx.saved_tensors
                    dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                    torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
                    return dr,dw,dk,dv,da,db
            def RUN_CUDA_RWKV7g(r,w,k,v,a,b):
                B,T,HC = r.shape
                r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
                return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HC)

        else:
            raise NotImplementedError("Unsupported precision for RWKV7 fine-tuning")
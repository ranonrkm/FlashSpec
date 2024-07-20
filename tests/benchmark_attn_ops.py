import sys
sys.path.append("..")
import torch
import torch._dynamo.config
import torch._inductor.config
from Engine.utils import custom_func, gqa_custom
import argparse 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=128)
parser.add_argument('--prefill_mqa', type=int, default=4096)
parser.add_argument('--prefill_gqa', type=int, default=16384)
args = parser.parse_args()

B = args.B
P1 = args.prefill_mqa
P2 = args.prefill_gqa
H = H_q = 32
H_k = 8
rep = H_q // H_k
D = 128

print(f"Running benchmark for B={B}, P1={P1}, P2={P2}, H={H}, H_q={H_q}, H_k={H_k}, D={D}")

for i in [1, 2, 4, 6]:

    # compile custom func and gqa_custom
    mqa_attn = torch.ops.mylib.custom_func
    gqa_attn = torch.ops.mylib.gqa_custom
    # mqa_attn = torch.compile(mqa_attn, mode="reduce-overhead", fullgraph=True)
    # gqa_attn = torch.compile(gqa_attn, mode="reduce-overhead", fullgraph=True)

    with torch.device('cuda'):
        print(f"dec len: {i}")
        q1 = torch.rand(B, i, H, D).half()
        K1 = torch.rand(B, P1+i, H, D).half()
        V1 = torch.rand(B, P1+i, H, D).half()
        k1 = torch.rand(B, i, H, D).half()
        v1 = torch.rand(B, i, H, D).half()
        seqlen1 = torch.full((B,), P1, dtype=torch.int32)

    # warm up
    for _ in range(100):
        mqa_attn(q1, K1, V1, k1, v1, seqlen1)

    prof = torch.profiler.profile()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    with prof:
        for _ in range(1000):
            mqa_attn(q1, K1, V1, k1, v1, seqlen1)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"Avg time taken for custom_func: {t2 - t1:.3f} ms")

    prof.export_chrome_trace(f"mqa_4k_H32_B{args.B}_dec{i}.json")

    del q1, K1, V1, k1, v1, seqlen1
    torch.cuda.empty_cache()

    with torch.device('cuda'):
        q2 = torch.rand(B, i, H_q, D).half()
        K2 = torch.rand(B, P2+i, H_k, D).half()
        V2 = torch.rand(B, P2+i, H_k, D).half()
        k2 = torch.rand(B, i, H_k, D).half()
        v2 = torch.rand(B, i, H_k, D).half()
        seqlen2 = torch.full((B,), P2, dtype=torch.int32)

    # warmup
    for _ in range(100):
        gqa_attn(q2, K2, V2, k2, v2, seqlen2)

    prof = torch.profiler.profile()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    with prof:
        for _ in range(1000):
            gqa_attn(q2, K2, V2, k2, v2, seqlen2)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"Avg time taken for gqa_custom: {t2 - t1:.3f} ms")

    prof.export_chrome_trace(f"gqa_16k_Hq32_Hk8_B{args.B}_dec{i}.json")

    del q2, K2, V2, k2, v2, seqlen2
    torch.cuda.empty_cache()

    with torch.device('cuda'):
        q3 = torch.rand(B, i*rep, H_k, D, dtype=torch.float16)
        K3 = torch.rand(B, P2+i, H_k, D, dtype=torch.float16)
        V3 = torch.rand(B, P2+i, H_k, D, dtype=torch.float16)
        seqlen3 = torch.full((B,), P2, dtype=torch.int32)

    # warm up
    for _ in range(100):
        mqa_attn(q3, K3, V3, k=None, v=None, cache_seqlens=None)

    prof = torch.profiler.profile()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    with prof:
        for _ in range(1000):
            mqa_attn(q3, K3, V3, k=None, v=None, cache_seqlens=None)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"Avg time taken for custom_func: {t2 - t1:.3f} ms")

    prof.export_chrome_trace(f"mqa_16k_H8_B{args.B}_dec{i*rep}.json")
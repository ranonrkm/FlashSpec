import time
import torch
import flashinfer

H = H_q = 32
H_k = 8
rep = H_q // H_k

P1 = 4096
P2 = P1 * rep
D = 128

for dec_len in [1, 2, 4, 6]:
    print(f"dec_len = {dec_len}")   

    # for MQA
    q1 = torch.randn(dec_len, H, D, dtype=torch.float16).to("cuda:0")
    k1 = torch.randn(P1, H, D, dtype=torch.float16).to("cuda:0")
    v1 = torch.randn(P1, H, D, dtype=torch.float16).to("cuda:0")

    for _ in range(100):
        o = flashinfer.single_prefill_with_kv_cache(q1, k1, v1, causal=True,
                allow_fp16_qk_reduction=True)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for _ in range(1000):
        o = flashinfer.single_prefill_with_kv_cache(q1, k1, v1, causal=True,
                allow_fp16_qk_reduction=True)
    
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"MQA: {t2 - t1:.3f} ms")
    del q1, k1, v1
    torch.cuda.empty_cache()

    # for GQA
    q2 = torch.randn(dec_len, H_q, D, dtype=torch.float16).to("cuda:0")
    k2 = torch.randn(P2, H_k, D, dtype=torch.float16).to("cuda:0")
    v2 = torch.randn(P2, H_k, D, dtype=torch.float16).to("cuda:0")

    for _ in range(100):
        o = flashinfer.single_prefill_with_kv_cache(q2, k2, v2, causal=True,
                allow_fp16_qk_reduction=True)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for _ in range(1000):
        o = flashinfer.single_prefill_with_kv_cache(q2, k2, v2, causal=True,
                allow_fp16_qk_reduction=True)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"GQA: {t2 - t1:.3f} ms")

    # for MQA based GQA
    q3 = q2.view(dec_len*rep, H_k, D).contiguous() # reshaped

    for _ in range(100):
        o = flashinfer.single_prefill_with_kv_cache(q2, k2, v2, causal=True,
                allow_fp16_qk_reduction=True)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for _ in range(1000):
        o = flashinfer.single_prefill_with_kv_cache(q2, k2, v2, causal=True,
                allow_fp16_qk_reduction=True)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"MQA based GQA: {t2 - t1:.3f} ms")
    

    
    
import torch
# import flashinfer
import time

total = 4096 * 32
bsz=256

for prefill in [4096, 8192, 16384, 32768]:
    num_heads = total // prefill
    print(f"Prefill {prefill}, num_heads {num_heads}")

    # attn of shape (num_heads, 4, prefill)    
    attn = torch.rand(bsz, num_heads, 4, prefill).cuda()

    for _ in range(100):
        torch.softmax(attn, dim=-1)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for _ in range(1000):
        torch.softmax(attn, dim=-1)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"Time: {t2 - t1:.3f} ms")

    del attn
    torch.cuda.empty_cache()
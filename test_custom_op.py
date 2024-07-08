import torch
from flash_attn import flash_attn_with_kvcache

torch.cuda.set_device(0)

with torch.device("cuda"):
    q = torch.randn((1, 2, 2, 4), dtype=torch.bfloat16)
    k_cache = torch.zeros((1, 5, 2, 4), dtype=torch.bfloat16)
    v_cache = torch.zeros((1, 5, 2, 4), dtype=torch.bfloat16)
    k = torch.randn((1, 2, 2, 4), dtype=torch.bfloat16)
    v = torch.randn((1, 2, 2, 4), dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([0], dtype=torch.int32)
    causal = True

torch.library.define(
    "mylib::custom_func",
    "(Tensor q, Tensor(a!) k_cache, Tensor(a!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens, bool causal) -> Tensor",
)

@torch.library.impl("mylib::custom_func", "cuda")
def custom_func(q, k_cache, v_cache, k, v, cache_seqlens, causal):
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=causal
    )

@torch.library.register_fake("mylib::custom_func")
def custom_func_abstract(q, k_cache, v_cache, k, v, cache_seqlens, causal):
    return torch.empty_like(q)

assert torch.allclose(
    flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=causal),
    torch.ops.mylib.custom_func(q, k_cache, v_cache, k, v, cache_seqlens, causal),
)

compiled_func = torch.compile(torch.ops.mylib.custom_func, fullgraph=True)

out = compiled_func(q, k_cache, v_cache, k, v, cache_seqlens, causal)
print(out)
print(k_cache)
print(v_cache)
print(cache_seqlens)


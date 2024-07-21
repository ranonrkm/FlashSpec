from typing import Optional
import math 
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from flash_attn import flash_attn_with_kvcache, flash_attn_func

import time

@torch.compile
def gqa_custom(q, k_cache, v_cache, k, v, attn_bias, scale_factor):
    '''
    q: [B, T*rep, H, D]
    k_cache: [B, S, H, D]
    v_cache: [B, S, H, D]
    k: [B, H, T, D]
    v: [B, H, T, D]
    '''
    #out_past, lse_past = torch.ops.mylib.custom_func(q, k_cache, v_cache)
    out = flash_attn_with_kvcache(q, k_cache, v_cache)
   # out_past = out_past.transpose(1, 2).contiguous()
    #lse_past = lse_past.unsqueeze(-1)

   # attn_weight = torch.einsum("bhtd,bhsd->bhts", q.transpose(1, 2).contiguous(), k) * scale_factor
   # attn_weight += attn_bias
   # max_new = attn_weight.max(dim=-1, keepdim=True).values
   # lse_new = torch.logsumexp(attn_weight - max_new, dim=-1, keepdim=True)
   # out_new = torch.exp(attn_weight - max_new - lse_new) @ v

   # sumexp_past = torch.exp(lse_past - max_new)
   # sumexp_new = torch.exp(lse_new)
   # sumexp_total =  sumexp_past + sumexp_new
   # out = (out_past * sumexp_past + out_new * sumexp_new) / sumexp_total

    return out

@torch.compile
def residual_attn(q, k, v, attn_bias, scale_factor):
    attn_weight = torch.einsum("bhtd,bhsd->bhts", q, k) * scale_factor + attn_bias
    max_attn = attn_weight.max(dim=-1, keepdim=True).values
    lse = torch.logsumexp(attn_weight - max_attn, dim=-1, keepdim=True)
    out = torch.exp(attn_weight - max_attn - lse) @ v
    return out, lse, max_attn

B=64
H_q=32
H_k=8
rep = H_q // H_k
D=128
T=2
S=16000
q = torch.rand(B, T, H_q, D).half().cuda()
k = torch.rand(B, T, H_k, D).half().cuda()
v = torch.rand(B, T, H_k, D).half().cuda()
q_reshaped = q.view(B, T, H_k, rep, D).transpose(2, 3).contiguous().view(B, -1, H_k, D)  # [B, T*r, H, D]
q_t = q_reshaped.transpose(1, 2).contiguous()  # [B, H, T*r, D]
k_t = k.transpose(1, 2).contiguous()  # [B, H, T, D]
v_t = v.transpose(1, 2).contiguous()  # [B, H, T, D]

k_cache = torch.rand(B, S, H_k, D).half().cuda()
v_cache = torch.rand(B, S, H_k, D).half().cuda()

cache_seqlens = torch.full((B,), S, dtype=torch.int32).cuda()

attn_bias = torch.zeros(T*rep, T, dtype=q.dtype, device=q.device)
attn_mask = torch.ones(T, T, device=q.device).triu(diagonal=0).repeat_interleave(rep, -1).transpose(0, 1)
attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
torch.compile(gqa_custom)

for _ in range(100):
    out = flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=True)
    residual_attn(q_t, k_t, v_t, attn_bias, 1.0 / math.sqrt(D))

torch.cuda.synchronize()
t1 = time.time()

for _ in range(1000):
    out_past = flash_attn_with_kvcache(q_reshaped, k_cache, v_cache)

torch.cuda.synchronize()
t2 = time.time()


for _ in range(1000):
    out_past = flash_attn_with_kvcache(q_reshaped, k_cache, v_cache)
    # out_new = F.scaled_dot_product_attention(q_t, k_t, v_t)
    out = gqa_custom(q_reshaped, k_cache, v_cache, k_t, v_t, attn_bias, 1.0 / math.sqrt(D))

torch.cuda.synchronize()
t3 = time.time()

print("flash_attn with MQA: ", (t2-t1), "ms")
print("flash attn with MQA + residule ", (t3-t2), "ms")
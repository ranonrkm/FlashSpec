from typing import Optional
import math 
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from flash_attn import flash_attn_with_kvcache


def hybrid_attn(q, k_cache, v_cache, k, v, cache_seqlens, causal):
    '''
    Args:
        q: [B, T, H, D]
        k_cache: [B, S, H, D]
        v_cache: [B, S, H, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
    '''
    B, T, H, D = q.size()
    assert T > 1, "only useful for T > 1"
    q_reshaped = q.view(-1, 1, H, D)  # [B*T, 1, H, D]
    # key and value of the first new token repeated for all new time steps
    k_reshaped = k[:, 0].unsqueeze(1).repeat(T, 1, 1, 1) # [B*T, 1, H, D]
    v_reshaped = v[:, 0].unsqueeze(1).repeat(T, 1, 1, 1) # [B*T, 1, H, D]
    out_flash, lse_flash = flash_attn_with_kvcache(q_reshaped, k_cache, v_cache, k_reshaped, v_reshaped, cache_seqlens, causal)
    out_flash = out_flash.view(B, T, H, D)
    lse_flash = lse_flash.view(B, T, H, 1)
    out_flash_rest = out_flash[:, 1:]  # [B, T-1, H, D]
    lse_flash_rest = lse_flash[:, 1:]  # [B, T-1, H, 1]

    q_final = q[:, 1:]  # [B, T-1, H, D]
    k_final = k[:, 1:]  # [B, T-1, H, D]
    v_final = v[:, 1:]  # [B, T-1, H, D]

    lse_final = torch.full_like(lse_flash_rest, float("-inf"))
    out_final = torch.zeros_like(out_flash_rest)

    scale_factor = 1. / math.sqrt(D)
    attn_bias = torch.zeros(T-1, T-1, dtype=query.dtype)
    temp_mask = torch.ones(T-1, T-1, dtype=torch.bool).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)

    attn_weight = q_final @ k_final.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    # compute the lse
    lse_final = torch.logsumexp(attn_weight, dim=-1, keepdim=True)
    out_final = torch.exp(attn_weight - lse_final) @ v_final    # [B, T-1, H, D]

    sumexp_flash_rest = torch.exp(lse_flash_rest)
    sumexp_final = torch.exp(lse_final)
    sumexp_total = sumexp_flash_rest + sumexp_final
    out_flash_rest = (out_flash_rest * sumexp_flash_rest + out_final * sumexp_final) / sumexp_total

    return out_flash


q = torch.rand(2, 4, 32, 128).cuda()
k = torch.rand(2, 4, 32//8, 128).cuda()
v = torch.rand(2, 4, 32//8, 128).cuda()

k_cache = torch.rand(2, 16, 32//8, 128).cuda()
v_cache = torch.rand(2, 16, 32//8, 128).cuda()

cache_seqlens = torch.LongTensor([16, 16]).cuda()

out_flash = flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, causal=True)

out_custom = hybrid_attn(q, k_cache, v_cache, k, v, cache_seqlens, causal=True)

assert torch.allclose(out_flash, out_custom, atol=1e-5)





import torch
import flashinfer
import time

num_layers = 1

num_qo_heads = 32

num_kv_heads = 4 # 8 # 32

head_dim = 128

max_num_pages = 256

page_size = 32768 #16384 #4096

# allocate 128MB workspace buffer

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

batch_size = 256
dec_len = 4
nnz_qo = batch_size * dec_len

qo_indptr = torch.arange(0, nnz_qo, dec_len, device="cuda:0").int()

paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")

paged_kv_indptr = torch.arange(0, batch_size+1).int().to("cuda:0")

# 1 <= paged_kv_last_page_len <= page_size
paged_kv_last_page_len = torch.full((batch_size,), page_size-dec_len, dtype=torch.int32, device="cuda:0")

q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")

kv_cache_at_layer = torch.randn(
    num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
)

# create auxiliary data structures for batch prefill attention

prefill_wrapper.begin_forward(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)

for _ in range(30):
    outputs = []

    for i in range(num_layers):
        q = q_at_layer[i]
        kv_cache = kv_cache_at_layer[i]
        # compute batch prefill attention, reuse auxiliary data structures
        o = prefill_wrapper.forward(
            q, kv_cache, causal=True
        )
        outputs.append(o)

torch.cuda.synchronize()
t1 = time.perf_counter()
for _ in range(100):
    outputs = []

    for i in range(num_layers):
        q = q_at_layer[i]
        kv_cache = kv_cache_at_layer[i]
        # compute batch prefill attention, reuse auxiliary data structures
        o = prefill_wrapper.forward(
            q, kv_cache, causal=True
        )
        outputs.append(o)

torch.cuda.synchronize()
t2 = time.perf_counter()

# clear auxiliary data structures
prefill_wrapper.end_forward()

print("prefill length: ", page_size)
print(f"Avg time taken for query head {num_qo_heads} and key-value head {num_kv_heads}: {(t2 - t1)*10:.3f} ms")

# # below is another example of creating custom mask for batch prefill attention

# mask_arr = []

# qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()

# kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()

# for i in range(batch_size):

#     mask_i = torch.tril(

#         torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),

#         diagonal=(kv_len[i] - qo_len[i]),

#     )

#     mask_arr.append(mask_i.flatten())


# mask = torch.cat(mask_arr, dim=0)

# prefill_wrapper.begin_forward(

#     qo_indptr,

#     paged_kv_indptr,

#     paged_kv_indices,

#     paged_kv_last_page_len,

#     num_qo_heads,

#     num_kv_heads,

#     head_dim,

#     page_size,

#     mask

# )

# outputs_custom_mask = []

# for i in range(num_layers):

#     q = q_at_layer[i]

#     kv_cache = kv_cache_at_layer[i]

#     # compute batch prefill attention, reuse auxiliary data structures

#     o_custom = prefill_wrapper.forward(

#         q, kv_cache

#     )

#     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)


# # clear auxiliary data structures

# prefill_wrapper.end_forward()
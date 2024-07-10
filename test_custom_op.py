# import torch

# BATCH_SIZE = 4
# gamma = 4
# tokens_buffer = torch.arange(BATCH_SIZE * (gamma + 1)).view(BATCH_SIZE, gamma + 1).to(torch.long).cuda()
# output = torch.zeros((BATCH_SIZE, 10), device='cuda').long()
# accept_nums = torch.tensor([3, 2, 5, 4], device='cuda').long()  # shape (BATCH_SIZE,)
# offset = torch.tensor([1, 2, 3, 4], device='cuda').long()  # shape (BATCH_SIZE,)

# # Create a mask for the positions to fill in the output tensor
# positions = torch.arange(10, device='cuda').view(1, -1).repeat(BATCH_SIZE, 1)
# mask = (positions < (offset + accept_nums).view(-1, 1)) & (positions >= offset.view(-1, 1))

# positions_buffer = torch.arange(gamma+1, device='cuda').view(1, -1).repeat(BATCH_SIZE, 1)
# mask_buffer = positions_buffer<accept_nums.view(-1,1)


# output[mask] = tokens_buffer[mask_buffer]
# print(tokens_buffer)
# print(output)



from transformers.models.llama.modeling_llama import(
    LlamaRMSNorm,
    LlamaConfig,
    PreTrainedModel,
    repeat_kv,
    ACT2FN
)

print(LlamaConfig.from_pretrained("JackFram/llama-68m"))
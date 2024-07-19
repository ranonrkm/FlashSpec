import os
import sys
import time
from pathlib import Path
sys.path.append("..")
import torch
import torch._dynamo.config
import torch._inductor.config
from tqdm import tqdm
from transformers import AutoTokenizer
from FlashSpec.Data.data_converter import convert_pg19_dataset
import argparse
from FlashSpec.Engine.backend import LMBackend

parser = argparse.ArgumentParser(description='Your CLI description.')
parser.add_argument('--checkpoint_path', type=Path, default=Path("/home/rsadhukh/vashisth/gpt-fast/checkpoints/meta-llama/Meta-Llama-3-8B/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--M', type=int, default=256, help='max len')
parser.add_argument('--P', type=int, default=128, help='prefix len')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
print(f"Using device={device}")

model_name="meta-llama/Meta-Llama-3-8B"
checkpoint_path = args.checkpoint_path
precision = torch.bfloat16
max_seq_length = args.M
max_batch_size = args.B
prefix_len = args.P

warm_up = 10
decode_len = 100
llm = LMBackend(dtype=precision, device=device)
llm.load_model(checkpoint_path, use_tp=False, rank_group=[-1], group = None)
if args.compile:
    llm.compile()
llm.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=prefix_len+decode_len+1)

pbar = tqdm(enumerate(dataset))
total_loss = 0
for i, sample in pbar:
    if i == 1:
        break
    input_ids = sample[0].unsqueeze(0).to(device)
    llm.encode(input_ids[:, :prefix_len])
    for j in range(decode_len):
        logits = llm.inference(input_ids[:, prefix_len+j:prefix_len+j+1])
        logits = logits.view(-1, logits.size(-1))
        loss = torch.nn.functional.cross_entropy(logits, input_ids[:, prefix_len+j+1].view(-1))
        pbar.set_postfix(loss=loss.item())
        total_loss += loss.item()
print(total_loss / decode_len)
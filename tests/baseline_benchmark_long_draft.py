import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from FlashSpec.Engine.utils import setup_seed, sample
from FlashSpec.Data.data_converter import convert_pg19_dataset
from transformers import LlamaTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from FlashSpec.Engine.backend_draft import LMBackend_Draft

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--checkpoint_path', type=Path, default=Path("/home/jianc2/FastHesse/checkpoints/JackFram/llama-68m/model.pth"), help='Model checkpoint path.')
parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--M', type=int, default=4096, help='Maximum length.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset', type=str, default="pg", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='Dataset start index.')
parser.add_argument('--end', type=int, default=200, help='Dataset end index.')
parser.add_argument('--top_p', type=float, default=0.9, help='Target sample top_p.')
parser.add_argument('--temperature', type=float, default=0.6, help='Target sample temperature.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')

args = parser.parse_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from FlashSpec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        # only print on rank 0
        print = lambda *args, **kwargs: None
setup_seed(args.seed)
print(f"Using device={DEVICE}")
MAX_LEN = args.M
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
checkpoint_path = args.checkpoint_path
engine = LMBackend_Draft(dtype=DTYPE, device=DEVICE)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=4096)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = len(dataloader)

total_time = 0.0
model_steps = 0
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    input_ids = batch[0][:, :4000].to(DEVICE)
    terminate = False
    output = input_ids.clone()
    logits = engine.encode(input_ids=input_ids)
    next_tokens = sample(logits=logits[:,-1], top_p=args.top_p, T=args.temperature)
    output = torch.cat((output, next_tokens),dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    while output.size(1)<args.M and terminate == False:
        input_ids=next_tokens.clone()
        logits = engine.inference(input_ids=input_ids)
        next_tokens = sample(logits=logits[:,-1], top_p=args.top_p, T=args.temperature)
        output = torch.cat((output, next_tokens),dim=-1)
        model_steps += 1
        if (next_tokens[:,-1] == 2)._is_any_true() or (next_tokens[:,-1] == 0)._is_any_true(): terminate = True
    torch.cuda.synchronize()
    t2=time.perf_counter()
    total_time += t2-t1
    print(f"Tokens per second :{total_time/model_steps}")
    if step == 0:
        total_time = 0.0
        model_steps = 0
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i, 4000:]))
    if use_tp:
        dist.barrier()
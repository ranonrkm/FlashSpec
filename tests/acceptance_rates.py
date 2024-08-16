import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from FlashSpec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from FlashSpec.Data.data_converter import convert_pg19_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from FlashSpec.tests.llama_quest import LlamaForCausalLM
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
import contextlib
from termcolor import colored

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='Model name.')
parser.add_argument('--budget', type=int, default=256, help='Dataset end index.')
parser.add_argument('--page_size', type=int, default=16, help='Page size.')

parser.add_argument('--prefix_len', type=int, default=4000, help='Prefix length')
parser.add_argument('--gen_len', type=int, default=64, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')

args = parser.parse_args()

setup_seed(args.seed)
DEVICE = "cuda:0"
print(f"Using device={DEVICE}")
MAX_LEN_TARGET = args.prefix_len + args.gen_len + 1
DTYPE = torch.bfloat16
BATCH_SIZE = 1

T=0.01

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")
repeats = 20
no_runs = int(BATCH_SIZE*repeats)
dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len) #, end=no_runs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(10, len(dataloader))

greedy = True
max_gen_toks = args.gen_len

model = LlamaForCausalLM.from_pretrained(args.model_name).to(DEVICE)
model.config.K = args.page_size
model.config.L = 1.
draft = LlamaForCausalLM.from_pretrained(args.model_name).to(DEVICE)
draft.config.K = args.page_size
draft.config.L = args.budget / model.config.max_position_embeddings

agg_acc_rate = 0.
total_samples = 0

pbar = tqdm(enumerate(dataloader), total=num_eval_steps)
for step, batch in pbar:
    if step >= num_eval_steps:
        break
    input_ids = batch[0].to(DEVICE)
    initial_len = input_ids.shape[1]
    tokens = input_ids.clone()    

    with torch.inference_mode():
        tokens = model.generate(tokens, max_new_tokens=max_gen_toks)    # [B, S]
        num_samples = tokens.shape[1] - initial_len
        
        target_logits :torch.Tensor= model(input_ids=tokens).logits

        # draft_logits  :torch.Tensor= draft(input_ids=tokens).logits
        # have to do it in autoregressive manner
        draft_logits = torch.zeros_like(target_logits)
        draft_logits[:, initial_len-1] = draft(input_ids=tokens[:, :initial_len]).logits[:, -1]
        past_key_values = None
        import pdb; pdb.set_trace()
        for i in range(initial_len, initial_len+num_samples-1):
            outputs = draft(input_ids=tokens[:, i:i+1], past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            draft_logits[:, i] = outputs.logits[:, -1]

        # target_logits = get_sampling_logits(target_logits, P, T, replicate=False) # TODO: do we need it for greedy decoding?
                
        target_logits = target_logits[...,initial_len:,:]   # [B, S, V]
        draft_logits = draft_logits[...,initial_len:,:]
        target_proba = torch.nn.functional.softmax(target_logits/T, dim=-1).unsqueeze(-1)
        draft_proba = torch.nn.functional.softmax(draft_logits/T, dim=-1).unsqueeze(-1)

        probas = torch.cat([target_proba, draft_proba], dim=-1)
        probas = torch.min(probas, dim=-1).values
        acceptance_rate = probas.sum(dim=-1)    # [B, S]
        
        total_acceptance_rate = acceptance_rate.sum(dim=-1) 
        total_acceptance_rate = total_acceptance_rate.cumsum_(dim=0)

        if args.printoutput:
            # print last 10 input tokens in red color, and print new tokens in green color
            print(colored(tokenizer.decode(input_ids[0, -10:], skip_special_tokens=True), "red"), colored(tokenizer.decode(tokens[0, -num_samples:], skip_special_tokens=True), "green"))
            
    agg_acc_rate += total_acceptance_rate[0] 
    total_samples += num_samples
    pbar.set_description("acc rate: {:.2f}".format(agg_acc_rate / total_samples))

print("acc rate: ", agg_acc_rate / total_samples)

    
    
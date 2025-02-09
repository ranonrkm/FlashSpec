import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from FlashSpec.Engine.utils import setup_seed, sample, cuda_graph_for_sampling_argmax_batch
from FlashSpec.Data.data_converter import convert_pg19_dataset
from transformers import LlamaTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from FlashSpec.Engine.backend import LMBackend
from FlashSpec.Engine.backend_draft import LMBackend_Draft

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-hf/model.pth"), help='model')
parser.add_argument('--target', type=Path, default=Path("checkpoints/meta-llama/Llama-2-70b-hf/model.pth"), help='target model')
parser.add_argument('--gamma', type=int, default=5, help='start')
parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--M', type=int, default=256, help='Maximum length.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='Dataset start index.')
parser.add_argument('--end', type=int, default=200, help='Dataset end index.')
parser.add_argument('--top_p', type=float, default=0.9, help='Target sample top_p.')
parser.add_argument('--temperature', type=float, default=0.2, help='Target sample temperature.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--draft_ranks', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')
parser.add_argument('--streamingllm_budget', type=int, default=256, help='Dataset end index.')


args = parser.parse_args()
assert args.M + args.gamma + 1 <= 4096

draft_tp = len(args.draft_ranks) > 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from FlashSpec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
draft_group = None
if use_tp:
    if draft_tp:
        rank, global_group, draft_group = init_dist(args.draft_ranks)
    else:
        rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        print = lambda *args, **kwargs: None

setup_seed(args.seed)
print(f"Using device={DEVICE}")
MAX_LEN = args.M + args.gamma +1
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
benchmark = args.benchmark
checkpoint_path = args.target
draft_checkpoint_path = args.model

target_dec_list = [args.gamma + 1]

engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_list=target_dec_list)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN)
target_sample = cuda_graph_for_sampling_argmax_batch(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, idx_len=args.gamma+1)

if not use_tp:
    draft = LMBackend_Draft(dtype=DTYPE, device=DEVICE, dec_list=[1,2])
    draft.load_model(draft_checkpoint_path, use_tp=False, rank_group=args.rank_group, group=global_group)
    if args.compile:
        draft.compile()
    draft.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN, kv_len=args.streamingllm_budget)
    draft_sample = {}
    for i in [1, 2]:
        draft_sample[i] = cuda_graph_for_sampling_argmax_batch(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, idx_len=i)
else:
    if rank in args.draft_ranks:
        draft = LMBackend_Draft(dtype=DTYPE, device=DEVICE, dec_list=[1,2])
        draft.load_model(draft_checkpoint_path, use_tp=draft_tp, rank_group=args.draft_ranks, group=draft_group)
        if args.compile:
            draft.compile()
        draft.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN, kv_len=args.streamingllm_budget)
        draft_sample = {}
        for i in [1, 2]:
            draft_sample[i] = cuda_graph_for_sampling_argmax_batch(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, idx_len=i)
    dist.barrier()

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=4096)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = len(dataloader)

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0


for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    input_ids = batch[0][:,:4000].to(DEVICE)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    output = torch.zeros(BATCH_SIZE, MAX_LEN, device=DEVICE).long()
    output[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]

    logits = engine.encode(input_ids=input_ids)
    tokens_buffer[:,:1] = sample(logits=logits[:,-1], top_p=args.top_p, T=args.temperature)
    logits = None
    if not use_tp:
        draft.encode(input_ids=input_ids)
    else:
        if rank in args.draft_ranks:
            draft.encode(input_ids=input_ids)
        dist.barrier()
    
    # next_double = False
    first = True
    double_buffer = torch.zeros((BATCH_SIZE, 2), device=DEVICE).long()
    cachelens_update = None


    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        # Draft speculation
        if not use_tp:
            for i in range(args.gamma):
                if i == 0:
                    if not first:
                        # The cachelens should increase 1 or 2
                        next_tokens = draft_sample[2](draft.inference(double_buffer, cachelen_update=cachelens_update))
                        tokens_buffer[:,i+1:i+2] = next_tokens.gather(1, cachelens_update.view(-1,1) - 1)
                        # next_double = False
                    else:
                        tokens_buffer[:,i+1:i+2] = draft_sample[1](draft.inference(tokens_buffer[:, i].view(-1,1)))
                        first = False
                    continue
                tokens_buffer[:,i+1:i+2] = draft_sample[1](draft.inference(tokens_buffer[:, i].view(-1,1)))
        else:
            if rank in args.draft_ranks:
                for i in range(args.gamma):
                    if i == 0:
                        if not first:
                            next_tokens = draft_sample[2](draft.inference(double_buffer,cachelen_update=cachelens_update))
                            tokens_buffer[:,i+1:i+2] = next_tokens.gather(1, cachelens_update.view(-1,1) - 1)
                            # next_double = False
                        else:
                            tokens_buffer[:,i+1:i+2] = draft_sample[1](draft.inference(tokens_buffer[:, i].view(-1,1)))
                            first = False
                        continue
                    tokens_buffer[:,i+1:i+2] = draft_sample[1](draft.inference(tokens_buffer[:, i].view(-1,1)))
            dist.broadcast(tokens_buffer, src=args.draft_ranks[0], group=global_group)

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time+=t2-t1

        # Target Verification
        target_logits = engine.inference(tokens_buffer)
        # target_tokens = sample(target_logits, args.top_p, args.temperature)
        target_tokens = target_sample(target_logits)
        target_logits = None
        target_steps+=1

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            target_time+=t3-t2

    # Verify loop
        bonus_tokens = torch.full((BATCH_SIZE, 1), 0, device=DEVICE).long()
        accept_nums = torch.full((BATCH_SIZE, 1), 1, device=DEVICE).long()
        accept_flags = torch.full((BATCH_SIZE, 1), True, device=DEVICE)
        for pos in range(args.gamma):
            target_token = target_tokens[:, pos]
            draft_token = tokens_buffer[:, pos+1]
            flag_accept = (target_token == draft_token).unsqueeze(1)
            # Ensure flags remain False once they have been set to False
            accept_flags = accept_flags & flag_accept
            # Only increase accept_nums where accept_flags are still True
            accept_nums += accept_flags.int()
            # Wether or not terminate
            condition = ((draft_token.unsqueeze(1) == 0) | (draft_token.unsqueeze(1) == 2)) & accept_flags
            if condition.any():
                terminal = True
            accept_flags = accept_flags & ~condition
        
        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]

        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten()
        max_limit = torch.full_like(accept_nums, args.gamma, device = DEVICE)
        limited_accept_nums = torch.min(accept_nums, max_limit)
        if not use_tp:
            draft.cachelens = draft.cachelens - args.gamma
            draft.cachelens += limited_accept_nums.flatten()
        else:
            if rank in args.draft_ranks:
                draft.cachelens = draft.cachelens - args.gamma
                draft.cachelens += limited_accept_nums.flatten()
        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == 2).any() or (bonus_tokens == 0).any():
            terminal = True
        num_nodes += accept_nums.flatten()

        # Check Number of Nodes + Bonus Token <= max_target_token
        if num_nodes.max() + 1 >= args.M:
            terminal = True
        # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
        if not terminal:
            tokens_buffer[:, :1] = bonus_tokens
            # if accept_nums.max() == args.gamma + 1:
                # next_double = True
                # double_buffer = torch.zeros((BATCH_SIZE, 2), device=DEVICE).long()
            mask = (accept_nums == (args.gamma + 1)).squeeze()
            double_buffer[:, 0] = torch.where(mask, tokens_buffer[:, -1], bonus_tokens[:, 0])
            double_buffer[:, 1] = torch.where(mask, bonus_tokens[:, 0], torch.zeros_like(bonus_tokens[:, 0]))
            non_zero_mask = double_buffer != 0
            cachelens_update = non_zero_mask.sum(dim=1).flatten()
        
        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3
        else:
            for i in range(BATCH_SIZE):
                output[i, num_nodes[i]] = bonus_tokens[i]
            num_nodes += 1
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3

    torch.cuda.synchronize()
    end=time.perf_counter()
    total_time += end-start
    num_gen_tokens += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i, :num_nodes[i]]))
    print("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / target_steps, num_gen_tokens, target_steps))
    if benchmark:
        print("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {}".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))
    if step < 10:
        total_time = 0.0
        num_gen_tokens = 0
        target_steps = 0
        if benchmark:
            draft_time = 0.0
            target_time = 0.0
            verify_loop = 0.0
    if use_tp:
        dist.barrier()
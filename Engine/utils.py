import torch
from FlashSpec.Engine.model import Transformer
import numpy as np
import random

def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    shape = logits.shape
    if top_p < 1.0:
                if len(shape)==3:
                    batch_size, seq_len, voc_size = logits.size()
                    logits = logits.reshape(-1, voc_size)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                logits[indices_to_remove] = float('-inf')
                if len(shape)==3:
                    logits = logits.reshape(batch_size, seq_len, voc_size)
    return logits

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

@torch.inference_mode()
def gather_kv(model, indices, offsets):
    indices_len = indices.shape[1]
    batch_size, num_heads, seq_length, head_dim = model.layers[0].attention.kv_cache.k_cache.shape
    indices = indices.unsqueeze(1).unsqueeze(3)
    indices = indices.expand(-1, num_heads, -1, head_dim)

    storage_ids = offsets+torch.arange(0, indices_len, dtype=torch.long, device=model.layers[0].attention.kv_cache.k_cache.device).unsqueeze(0)
    storage_ids = storage_ids.unsqueeze(1).unsqueeze(3)
    storage_ids = storage_ids.expand(-1, num_heads, -1, head_dim)
    
    for b in model.layers:
        # Gather all k_cache and v_cache values at once
        new_k_cache = torch.gather(b.attention.kv_cache.k_cache, 2, indices)
        new_v_cache = torch.gather(b.attention.kv_cache.v_cache, 2, indices)
        # Perform batch index_copy_
        b.attention.kv_cache.k_cache.scatter_(src = new_k_cache, dim = 2, index = storage_ids)
        b.attention.kv_cache.v_cache.scatter_(src = new_v_cache, dim = 2, index = storage_ids)

def cuda_graph_for_gather_kv(
                device="cuda:0", 
                batch_size=1, max_len=7, model=None, 
                n_warmups=3, mempool=None):
    
    static_offsets = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
    static_indices = torch.arange(0,max_len, dtype=torch.long, device=device).repeat(batch_size,1)
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            gather_kv(model, static_indices, static_offsets)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        gather_kv(model, static_indices, static_offsets)
    def run(indices, offsets):
        static_offsets.copy_(offsets)
        static_indices.copy_(indices)
        graph.replay()
    return run

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def load_model(checkpoint_path, device, precision, use_tp, rank_group=None, group=None):

    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from FastHesse.Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()
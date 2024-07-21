import torch
from datasets import load_dataset
import os
import json
from torch.utils.data import TensorDataset, IterableDataset
from tqdm import tqdm

def convert_c4_dataset(tokenizer, file_path):
    dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            input_ids = torch.Tensor(examples['input_ids'])
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids,
                "labels": labels
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_tokens'])
    dataset.set_format(type='torch', columns=['input_ids', "labels"])
    return dataset

def convert_wiki_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_cnn_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["article"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_pg19_dataset(tokenizer, seq_len = 4096, end = 20):
    datasetparent = "Data/pg19/"
    d_files = os.listdir(datasetparent)
    dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
    tokenized_prompts = []
    for i in tqdm(range(0,20)):
        prompt = dataset[i]['text']
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:,8000:]
        tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]
        
        for i in range(len(tokenized_prompt)):
             tokenized_prompt[i][:, 0] = 1
             tokenized_prompts.append(tokenized_prompt[i])
    data = torch.cat(tokenized_prompts, dim=0).repeat(end,1)
    return TensorDataset(data)

class LongBenchDataset(IterableDataset):
    def __init__(self, tokenizer, task, seq_len = 4096):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset = load_dataset("THUDM/LongBench", task, split='test', streaming=True)
        self.data_iter = iter(self.dataset)
        prompt = json.load(open(f"Data/json/longbench_prompt.json", "r"))
        self.prompt_format = prompt[task]

    def __iter__(self):
        while True:
            try:
                sample = next(self.data_iter)
            except StopIteration:
                print("Restarting data iterator")
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
            prompt = self.prompt_format.format(**sample)
            prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
            tokenized_prompt = self.tokenizer.encode(prompt, truncation=False)
            if len(tokenized_prompt) >= self.seq_len:
                if len(tokenized_prompt) > self.seq_len:
                    half = self.seq_len // 2
                    prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+self.tokenizer.decode(tokenized_prompt[-self.seq_len+half:], skip_special_tokens=True)
                tokenized_prompt = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
                if tokenized_prompt.shape[0] == self.seq_len:
                    yield tokenized_prompt     

# if __name__ == "__main__":
#     from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
#     from torch.utils.data import DataLoader, TensorDataset
#     from tqdm import tqdm
#     tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     tokenizer.pad_token = tokenizer.eos_token
#     dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=4096)

#     dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
#     num_eval_steps = len(dataloader)
#     for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
#         input_ids = batch[0]
    
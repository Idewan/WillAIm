import os
import pickle
import sys
import torch
import torch.nn as nn

from torch.optim import Adam as AdamW
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from transformers import GPTNeoModel, GPTNeoForCausalLM
from tqdm import tqdm


class ClipPoemDataset(Dataset):

    def __len__(self):
        """
        
        """
        return len(self.poems_tokens)
    
    def __getitem__(self, item):
        tokens, mask = self.pad_tokens(item)
#        print(len(self.prefixes))
 #       print(len(self.poems2embedding))
  #      print(item)
        prefix = self.prefixes[self.poems2embedding[item]]

        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        
        return tokens, mask, prefix
    
    def __init__(self, data_path, prefix_length, gpt2_type="gpt2", normalize_prefix=False):
        """
        
        """
        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type,
        #                                 bos_token="<|startoftext|>",
        #                                 eos_token="<|endoftext|>",
        #                                 pad_token="<|pad|>")
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        self.prefixes = data['clip_embedding']
        poems = data['poems']

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", "rb") as f:
                self.poems_tokens, self.poems2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.poems_tokens = []
            self.poems2embedding = []
            max_seq_len = 0

            for poem in poems:
                self.poems_tokens.append(torch.tensor(self.tokenizer.encode(poem['poem']), dtype=torch.int64))
                self.poems2embedding.append(poem["clip_embedding"])
                max_seq_len = max(max_seq_len, self.poems_tokens[-1].shape[0])
            
            with open(f"{data_path[:-4]}_tokens.pkl", "wb") as f:
                pickle.dump([self.poems_tokens, self.poems2embedding, max_seq_len], f)

        #Compute the desired max length of the sequence
        #Take the minimum of the (mean + 10std above mean) and the max length of a sequence
        all_len = torch.tensor([len(self.poems_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def pad_tokens(self, item):
        """
        
        """
        tokens = self.poems_tokens[item]
        len_tokens = tokens.shape[0]
        
        if len_tokens < self.max_seq_len:
            padding = self.max_seq_len - len_tokens

            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)-1))
            self.poems_tokens[item] = tokens
        elif len_tokens > self.max_seq_len:

            tokens = tokens[:self.max_seq_len]
            self.poems_tokens[item] = tokens
        
        mask = self.get_mask(tokens)

        return tokens, mask
        
    def get_mask(self, tokens):
        """

        """
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)

        return mask

class MLP(nn.Module):

    def forward(self, x):
        """
        
        """
        return self.model(x)
    
    def __init__(self, sizes, bias=True):
        """
        
        """
        super(MLP, self).__init__()

        layers = []

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

class GPTMLPModel(nn.Module):
    
    def __init__(self, prefix_length, prefix_size=512):
        """
        
        """
        super().__init__()

        self.device = torch.device('cuda:0')
        self.prefix_length = prefix_length
        # self.gpt = GPTNeoForCausalLM.from_pretrained("news-gpt-neo-1.3B-keywords-line-by-line-reverse/checkpoint-15000").cuda()
        
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        
        self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                self.gpt_embedding_size * prefix_length))

    def forward(self, tokens, prefix, mask=None, labels=None):
        """
        
        """
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0])
            labels = torch.cat((dummy_token,tokens), dim=1)
        
        return self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)


    def get_dummy_token(self, batch_size):
        """
            GPT takes as input some input labels, hence if none are required we create blanks.
        """
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=self.device)
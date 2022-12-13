import os
import pickle
import sys
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

class ClipPoemDataset(Dataset):

    def __len__(self):
        """
        
        """
        return len(self.poems_tokens)
    
    def __getitem__(self, item):
        tokens, mask = self.pad_tokens(item)
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
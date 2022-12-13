import os
import pickle
import sys
import torch
import torch.nn as nn

from torch.optim import Adam as AdamW
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncodeLayer

from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from transformers import GPTNeoModel, GPTNeoForCausalLM
from tqdm import tqdm

class Transformer(nn.Module):
    def __init__(self, ntoken, ninp, nheadm, ninhid, nlayers, dropout=0.5, prefix_size=512):


class GPTTransformerModel(nn.Module):

    def __init__(self, prefix_length, clip_length, prefix_size=512):
        """

        """
        super(GPTTransformerModel, self).__init__()

        self.device = torch.device('cuda:0')
        self.prefix_length = prefix_length

        # self.gpt = GPTNeoForCausalLM.from_pretrained("news-gpt-neo-1.3B-keywords-line-by-line-reverse/checkpoint-15000").cuda()
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.clip_project = Transformer(prefix_size, self.gpt_embedding_size, prefix_length, clip_length,
                                        num_layers)
    
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



    
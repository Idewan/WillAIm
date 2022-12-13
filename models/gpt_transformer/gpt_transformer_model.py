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

class GPTTransformerModel(nn.Module):

    def __init__(self, prefix_length, prefix_size=512):
        """

        """
    
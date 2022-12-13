import sys
import os
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from gpt_mlp.gpt_clip_model import GPTMLPModel
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from transformers import GPT2Tokenizer
from PIL import Image

import torch
import numpy as np
import clip

# class NucleusSampling(Sampler):

#     def __init__(self, p: float, sampler: Sampler):
#         self.p = p
#         self.sampler = sampler
    
#     def __call__(self, logits: torch.Tensor):
#         probs = torch.nn.softmax(logits)

#         #We want to find the sum of top-v words 
#         #such that the sum of their probability is
#         # >= p
#         sorted_probabilities, indices = torch.sort(probs, dim=-1, descending=True) 
#         cumsum = torch.cumsum(sorted_probabilities, dim=-1)

#         nucleus = cumsum < self.p

#         nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

#         sorted_log_probs = torch.log(sorted_probabilities)
#         sorted_log_probs[~nucleus] = float('-inf')

#         sampled_sorted_indexes = self.sampler(sorted_log_probs)

#         res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

#         return res.squeeze(-1)

class Decoding:

    def __init__(self, model_path):
        #Preprocessing
        self.device = torch.device('cuda:0')
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

        #Model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPTMLPModel(prefix_length=10)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
    
    def predict(self, image):
        """
        
        """
        image_pp = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image_pp).to(self.device).float()

        prefix_embed = self.model.clip_project(prefix).reshape(1, 10, -1)

        return self.generate_nucleus(embed=prefix_embed)

    def generate_nucleus(self, p: int = 0.9, embed=None,
                        entry_length=77, stop_token: str = "."):
        #Initialize
        self.model.eval()
        # self.nucleus_sampling = NucleusSampling(p=p)
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")

        tokens = None

        with torch.no_grad():
            if embed is not None:
                generated = embed
                print("hey")
            else:
                generated = self.model.gpt.transformer.wte(tokens)

            for _ in range(entry_length):
                
                preds = self.model.gpt(inputs_embeds=generated)
                logits = preds.logits

                logits = logits[:, -1, :] / 1
                
                # next_token = self.nucleus_sampling(logits)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > p

                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                print(logits.shape)
                # next_tokens = torch.argmax(logits, -1).unsqueeze(0)
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                # next_token = self.nucleus_sampling(logits)
                print(next_token)
                
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                    
                next_token_embed = torch.reshape(torch.squeeze(self.model.gpt.transformer.wte(next_token), 0),(1,1,768))
                generated = torch.cat((generated, next_token_embed), dim=1)

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = self.tokenizer.decode(output_list)
        
        return output_text
    

            
                





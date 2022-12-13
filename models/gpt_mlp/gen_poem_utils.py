import sys
import os
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from gpt_mlp.gpt_clip_model import GPTMLPModel
from torch import Tensor
from torch.nn import functional as F
from transformers import GPT2Tokenizer
from PIL import Image

import torch
import numpy as np
import clip

class GenPoemUtils():

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

        return self.generate_beam(embed=prefix_embed)

    def generate_beam(self, beam_size: int = 5, prompt=None, embed=None,
                    entry_length=67, temperature=0.85,   stop_token: str = ".",):
        """
        
        """
        self.model.eval()
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(self.model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(self.tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = self.model.gpt.transformer.wte(tokens)
            for _ in range(entry_length):
                outputs = self.model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.model.gpt.transformer.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [
            self.tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts


    # def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), min_tokens_to_keep = 1, return_index = False):
    #     """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    #         Args:
    #             logits: logits distribution shape (vocabulary size)
    #             top_k >0: keep only top k tokens with highest probability (top-k filtering).
    #             top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    #                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    #         Taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    #     """
    #     if top_k > 0:
    #             top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
    #             # Remove all tokens with a probability less than the last token of the top-k
    #             indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    #             indices_keep = logits >= torch.topk(logits, top_k)[0][..., -1, None]
    #             indices_keep = indices_keep[0].tolist()
    #             indices_keep = [i for i,x in enumerate(indices_keep) if x == True]
    #             logits[indices_to_remove] = filter_value
    #     if top_p < 1.0:
    #         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #         # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    #         sorted_indices_to_remove = cumulative_probs > top_p
    #         if min_tokens_to_keep > 1:
    #             # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
    #             sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    #         # Shift the indices to the right to keep also the first token above the threshold
    #         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #         sorted_indices_to_remove[..., 0] = 0

    #         # scatter sorted tensors to original indexing
    #         indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    #         logits[indices_to_remove] = filter_value
            
    #     if return_index == True:
    #         return logits, indices_keep
    #     return logits

    # def generate_next_word(self, embed, temperature = 0.85, topk = 100, device = 'cuda:0'):
    #     """

    #     """
    #     current_word = 0
    #     for _ in range(10):
    #         outputs1 = self.model.gpt(input_embeds=embed)
    #         next_token_logits1 = outputs1[0][:, -1, :]
    #         next_token_logits1 = self.top_k_top_p_filtering(next_token_logits1, top_k=topk)
    #         logit_zeros = torch.zeros(len(next_token_logits1)).cuda()

    #         next_token_logits = next_token_logits1 * temperature
    #         probs = F.softmax(next_token_logits, dim=-1)
    #         next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    #         unfinished_sents = torch.ones(1, dtype=torch.long).cuda()
    #         tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

    #         if self.tokenizer.eos_token_id in next_tokens[0]:
    #             embed = torch.cat([embed, tokens_to_add.unsqueeze(-1)], dim=-1)
    #             return '', True

    #         if self.tokenizer.decode(tokens_to_add[0])[0] == ' ':
    #             if current_word == 1:
    #                 return self.tokenizer.decode(embed[0]).split()[-1], False
    #             current_word += 1
    #         embed = torch.cat([embed, tokens_to_add.unsqueeze(-1)], dim=-1)
    #     return None
    

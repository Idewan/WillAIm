import torch
import toch.nn as nn

class NucleusSampling(Sampler):

    def __init__(self, p: float, sampler: Sampler):
        self.p = p
        self.sampler = sampler
    
    def __call__(self, logits: torch.Tensor):
        probs = torch.nn.softmax(logits)

        #We want to find the sum of top-v words 
        #such that the sum of their probability is
        # >= p
        sorted_probabilities, indices = torch.sort(probs, dim=-1, descending=True) 
        cumsum = torch.cumsum(sorted_probabilities, dim=-1)

        nucleus = cumsum < self.p

        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

        sorted_log_probs = torch.log(sorted_probabilities)
        sorted_log_probs[~nucleus] = float('-inf')

        sampled_sorted_indexes = self.sampler(sorted_log_probs)

        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

        return res.squeeze(-1)

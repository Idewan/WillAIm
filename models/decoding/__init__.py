import torch


class Sampler:
    """
    ### Sampler base class
    """
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        ### Sample from logits
        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        """
        raise NotImplementedError()
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    r"""This criterion computes the Bayesian Personalized Ranking (BPR) loss between the positive scores and the negative scores.

    Args:
        ``alpha`` (``float``, optional): The weight for the positive scores in the BPR loss. Defaults to ``1.0``.
        ``beta`` (``float``, optional): The weight for the negative scores in the BPR loss. Defaults to ``1.0``.
        ``activation`` (``str``, optional): The activation function to use can be one of ``"sigmoid_then_log"``, ``"softplus"``. Defaults to ``"sigmoid_then_log"``.
    """
    def __init__(self, alpha:float=1.0, beta:float=1.0, activation:str="sigmoid_then_log"):
        super().__init__()
        assert activation in ("sigmoid_then_log", "softplus",), "activation function of BPRLoss must be sigmoid_then_log or softplus."
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        r"""The forward function of BPRLoss.
        
        Args:
            ``pos_scores`` (``torch.Tensor``): The positive scores.
            ``neg_scores`` (``torch.Tensor``): The negative scores.
        """
        if self.activation == "sigmoid_then_log":
            loss = -(self.alpha * pos_scores - self.beta * neg_scores).sigmoid().log()
        elif self.activation == "softplus":
            loss = F.softplus(self.beta * neg_scores - self.alpha * pos_scores)
        else:
            raise NotImplementedError
        return loss.mean()

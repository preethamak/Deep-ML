import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    
    sc = torch.as_tensor(scores)

    log_sc = torch.nn.functional.log_softmax(sc, dim = 0)

    return log_sc

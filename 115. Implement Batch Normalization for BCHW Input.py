import numpy as np

def batch_norm_bchw(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Batch Normalization for BCHW input.
    
    Args:
        X: Input tensor of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        epsilon: Small constant for numerical stability
    
    Returns:
        Normalized and scaled output of shape (B, C, H, W)
    """
    # Compute mean and variance per channel
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
    var = np.var(X, axis=(0, 2, 3), keepdims=True)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    
    # Scale and shift
    out = gamma * X_norm + beta
    
    return out

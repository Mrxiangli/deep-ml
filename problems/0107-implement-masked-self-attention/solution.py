import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return X @ W_q, X @ W_k, X @ W_v 

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # Your code here
    weights = Q @ K.T / np.sqrt(K.shape[-1])
    masked_attention = weights + mask

    def softmax(x):
        shift = x - np.max(x, axis = -1,  keepdims = True)
        return np.exp(shift) / np.sum(np.exp(shift), axis = -1, keepdims = True)
    
    return softmax(masked_attention) @ V
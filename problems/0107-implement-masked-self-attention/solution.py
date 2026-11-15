import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x):
    shift = x - np.max(x,axis=-1, keepdims=True)

    return np.exp(shift) / np.sum(np.exp(shift),axis=-1, keepdims=True) 

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # n = Q.shape[0]
    # mask = np.zeros((n, n))
    # mask[np.triu_indices(n, k=1)] = -np.inf
    d_k = K.shape[-1]
    

    return softmax(Q @ K.T /np.sqrt(d_k) + mask) @ V
    
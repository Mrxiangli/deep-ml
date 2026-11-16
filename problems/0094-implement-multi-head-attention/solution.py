import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return (Q, K, V)

def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)
 
def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = K.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d_k)) @ V 

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """

    seq_len, d_model = Q.shape

    d_head = d_model // n_heads

    output = []

    for h in range(n_heads):
        cols = slice(h*d_head,(h+1)*d_head)
        q,k,v = Q[:, cols], K[:,cols], V[:, cols]
        res = self_attention(q,k,v)
        output.append(res)
    
    return np.concatenate(output, axis=-1)

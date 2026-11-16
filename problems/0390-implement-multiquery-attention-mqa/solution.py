import numpy as np

def softmax(x):
    shift = x - np.max(x, axis = -1, keepdims = True)
    return np.exp(shift) / np.sum(np.exp(shift), axis = -1, keepdims=True)

def self_attention(q,k,v):
    return softmax(q @ k.T/np.sqrt(k.shape[-1])) @ v

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """

    outputs = []

    K = X @ W_key
    V = X @ W_value

    for W_q in W_queries:
        Q = X @ W_q
        outputs.append(self_attention(Q, K, V))

    combined = np.concatenate(outputs, axis=-1)
    return np.round(combined @ W_out, 4)
    
    
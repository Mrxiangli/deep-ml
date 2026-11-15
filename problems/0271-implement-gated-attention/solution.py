import numpy as np

def sigmoid(x):
    y = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + y), y / (1 + y))

def softmax(x):
    shift = x - np.max(x, axis = -1, keepdims=True)
    return np.exp(shift) / np.sum(np.exp(shift), axis = -1, keepdims=True)


def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = K.shape[-1]
    score = softmax(Q @ K.T/ np.sqrt(d_k)) @ V 
    return sigmoid(X @ W_g) * score

    
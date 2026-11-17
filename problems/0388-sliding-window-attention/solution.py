import numpy as np

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """

    def softmax(x):
        shift = x - np.max(x, axis = -1, keepdims=True)
        return np.exp(shift) / np.sum(np.exp(shift), axis = -1, keepdims=True)

    seq_len, d_k = K.shape[0], K.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)

    for i in range(seq_len):
        s = max(0, i - window_size)
        e = min(seq_len, i+window_size+1)

        scores[i,:s] = -np.inf
        scores[i,e:] = -np.inf
    
    output = softmax(scores) @ V
    return np.round(output,4)



    
import numpy as np
def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    seq_len, d_k = K.shape[0], K.shape[1]

    if not scale_factor:
        scale_factor = np.sqrt(d_k)
    
    output = np.empty((seq_len, V.shape[1]), dtype=float)

    for i in range(seq_len):
        s = max(0, i - window_size)
        e = min(seq_len, i + window_size + 1)

        k = K[s:e]
        v = V[s:e]

        scores = Q[i] @ k.T / scale_factor

        weights = np.exp(scores - np.max(scores, axis = -1, keepdims=True)) / np.sum(np.exp(scores - np.max(scores, axis = -1, keepdims=True)), axis = -1, keepdims = True)

        output[i] = weights @ V[s:e]
    
    return output



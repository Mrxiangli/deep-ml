import numpy as np


def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                           block_size: int = 2) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    seq_len, d_model = K.shape
    scale =  np.sqrt(d_model)
    output = np.empty((seq_len, V.shape[1]), dtype=float)

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start+block_size,seq_len)
        q = Q[q_start:q_end]
        n_queries = len(q)

        m = np.full((n_queries,1), -np.inf)
        l = np.zeros((n_queries,1))
        A = np.zeros((n_queries, V.shape[-1]))

        for k_start in range(0, seq_len, block_size):
            k_end = min(k_start+block_size, seq_len)
            k = K[k_start:k_end]
            v = V[k_start:k_end]

            s = q @ k.T / scale
            m_new = np.maximum(m, np.max(s, axis=-1, keepdims=True))
            l = np.exp(m-m_new) * l + np.sum(np.exp(s-m_new),axis = -1, keepdims=True)
            A = np.exp(m-m_new) * A + np.exp(s-m_new) @ v
            m = m_new
        output[q_start:q_end] = A/l
    
    return output


   
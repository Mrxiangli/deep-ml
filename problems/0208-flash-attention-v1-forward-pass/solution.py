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
    seq_len, d_model =  Q.shape

    output = np.empty((seq_len, d_model))
    scale = np.sqrt(d_model)

    for q_start in range(0, seq_len, block_size):

        q_end = min(q_start+block_size, seq_len)
        q_block = Q[q_start:q_end]
        n_queries = len(q_block)

        m = np.full((n_queries, 1), -np.inf)
        l = np.zeros((n_queries, 1))
        A = np.zeros((n_queries, V.shape[1]))

        for k_start in range(0, seq_len, block_size):
            k_end = min(k_start + block_size, seq_len)
            k_block = K[k_start:k_end]
            v_block = V[k_start:k_end]

            scores = (q_block @ k_block.T) / scale
            m_new = np.maximum(m, np.max(scores, axis=-1, keepdims=True))

            correction = np.exp(m - m_new)
            weights = np.exp(scores - m_new)  # Unnormalized

            l = correction * l + weights.sum(axis=-1, keepdims=True)
            A = correction * A + weights @ v_block
            m = m_new

        output[q_start:q_end] = A / l

    return output

    
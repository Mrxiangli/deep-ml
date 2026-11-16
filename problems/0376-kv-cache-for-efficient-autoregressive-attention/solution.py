import numpy as np

def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims = True)
    return np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims = True)

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    Q =  x_new @ W_Q

    k_new = (x_new @ W_K)[None, :]  # (1, d_k)
    v_new = (x_new @ W_V)[None, :]  # (1, d_v)

    if cache is None:
        K_cache, V_cache = k_new, v_new
    else:
        K_cache = np.concatenate((cache[0], k_new), axis=0)
        V_cache = np.concatenate((cache[1], v_new), axis=0)

    d_k = W_K.shape[-1]

    output = softmax(Q @ K_cache.T  / np.sqrt(d_k)) @ V_cache
    
    return (output,(K_cache, V_cache))




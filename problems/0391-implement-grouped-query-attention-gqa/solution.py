import numpy as np

def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims = True)
    return np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

def self_attention(q,k,v):
    return softmax(q @ k.swapaxes(-1, -2)  / np.sqrt(k.shape[-1])) @ v

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    head_d = Q.shape[-1] // num_heads
    B, seq_len = Q.shape[0], Q.shape[1] 
    Q = Q.reshape(B,  seq_len, num_heads, head_d).transpose(0,2,1,3)
    K = K.reshape(B,  seq_len, num_kv_heads, head_d).transpose(0,2,1,3)
    V = V.reshape(B,  seq_len, num_kv_heads, head_d).transpose(0,2,1,3)
    repeats = num_heads // num_kv_heads
    K_expand = np.repeat(K, repeats, axis = 1)
    V_expand = np.repeat(V, repeats, axis = 1)
    outputs = []
    for h in range(num_heads):
        q = Q[:,h]
        k = K_expand[:,h]
        v = V_expand[:,h]
        output = self_attention(q,k,v)
        outputs.append(output)
    res = np.concatenate(outputs, axis = -1)
    return res



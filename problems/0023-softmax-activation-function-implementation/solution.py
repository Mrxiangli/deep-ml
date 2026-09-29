import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    # exp(x) / sum(exp(x))
    scores = np.array(scores)
    # for stability implementation
    shifted = scores - np.max(scores)

    res = np.exp(shifted)/np.sum(np.exp(shifted))
    return res

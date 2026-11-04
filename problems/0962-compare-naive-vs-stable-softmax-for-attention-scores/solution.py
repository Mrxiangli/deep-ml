import numpy as np

def compare_softmax(scores: list) -> dict:
    scores = np.array(scores, dtype=np.float64)

    naive = np.exp(scores) / np.sum(np.exp(scores))

    shifted = scores - np.max(scores)
    stable = np.exp(shifted) / np.sum(np.exp(shifted))

    if not np.all(np.isfinite(naive)):
        naive = np.full_like(naive, np.nan)
        max_abs_diff = float('nan')
    else:
        max_abs_diff = round(float(np.max(np.abs(naive - stable))), 6)


    return {
        'naive': np.round(naive, decimals=6).tolist(),
        'stable': np.round(stable, decimals=6).tolist(),
        'max_abs_diff': max_abs_diff
    }
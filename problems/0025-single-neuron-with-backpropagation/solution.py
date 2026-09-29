import numpy as np

def train_neuron(
    features: list[list[float]],
    labels: list[int],
    initial_weights: list[float],
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[list[float], float, list[float]]:

    # 1. Prepare inputs as floats
    x = np.array(features, dtype=float)                  # Shape: (N, d)
    y = np.array(labels, dtype=float)                    # Shape: (N,)
    weights = np.array(initial_weights, dtype=float).copy() # Shape: (d,)
    bias = float(initial_bias)

    N = x.shape[0]
    mse_values = []

    for _ in range(epochs):
        # 2. Forward pass with Sigmoid activation
        z = np.dot(x, weights) + bias                    # Linear sum (N,)
        y_hat = 1 / (1 + np.exp(-z))                     # Sigmoid activation (N,)

        # 3. Record MSE Loss BEFORE update
        mse = np.mean((y_hat - y) ** 2)
        mse_values.append(round(float(mse), 4))

        # 4. Compute error delta (including sigmoid derivative term)
        # dL/dz = (2/N) * (y_hat - y) * y_hat * (1 - y_hat)
        delta = (2 / N) * (y_hat - y) * y_hat * (1 - y_hat)

        # 5. Compute Gradients
        w_grad = np.dot(x.T, delta)                      # Shape: (d,)
        b_grad = np.sum(delta)                           # Scalar

        # 6. Apply Gradient Descent Update
        weights -= learning_rate * w_grad
        bias -= learning_rate * b_grad

    # Convert back to standard Python types rounded to 4 decimals
    updated_weights = [round(float(w), 4) for w in weights]
    updated_bias = round(float(bias), 4)

    return updated_weights, updated_bias, mse_values
"""Task 1 - Neural Network Fundamentals (NumPy XOR network, activations, autograd, loss curves)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- activation + loss functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())          # subtract max for numerical stability
    return e / e.sum()


def mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)


def cross_entropy(y_true, y_prob, eps=1e-12):
    """categorical cross-entropy; y_true one-hot, y_prob softmax output"""
    y_prob = np.clip(y_prob, eps, 1)
    return -np.sum(np.asarray(y_true) * np.log(y_prob))


def bce(y, p, eps=1e-12):
    """binary cross-entropy used to train the XOR network"""
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ----------------------------------------------------------------- 2-4-1 network in NumPy
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)


def train_xor(lr=0.5, epochs=5000, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, 4)) * np.sqrt(2 / 2)   # He init (ReLU)
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, (4, 1)) * np.sqrt(1 / 4)
    b2 = np.zeros((1, 1))
    losses = []
    for _ in range(epochs):
        # forward pass
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        losses.append(bce(y, a2))
        # backward pass (sigmoid + BCE -> dz2 = a2 - y)
        n = X.shape[0]
        dz2 = (a2 - y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        # gradient descent step
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    preds = sigmoid(relu(X @ W1 + b1) @ W2 + b2).ravel()
    return losses, preds


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("=== Activations ===")
    print("relu([-1,0,2])   =", relu(np.array([-1, 0, 2])))
    print("sigmoid([-1,0,2])=", sigmoid(np.array([-1, 0, 2])))
    probs = softmax([2.0, 1.0, 0.1])
    print("softmax([2.0, 1.0, 0.1]) =", probs, "(sum =", round(probs.sum(), 3), ")")
    print("mse([1,0],[0.9,0.2])           =", round(mse([1, 0], [0.9, 0.2]), 4))
    print("cross_entropy([1,0,0], probs)  =", round(cross_entropy([1, 0, 0], probs), 4))

    print("\n=== Gradient by hand vs autograd ===")
    print("L = (w*x + b - t)^2 ; dL/dw = 2(w*x+b-t)*x = 2(2-10)*2 = -32 ; dL/db = 2(w*x+b-t) = -16")
    try:
        import torch
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        loss = (w * 2 + b - 10) ** 2
        loss.backward()
        print(f"Autograd: w.grad = {w.grad.item()}, b.grad = {b.grad.item()}")
    except ImportError:
        print("PyTorch not installed - run: pip install torch")

    print("\n=== XOR training (2-4-1, ReLU + Sigmoid, BCE, 5000 epochs) ===")
    results = {}
    for lr in (0.5, 0.01, 5):
        losses, preds = train_xor(lr=lr)
        results[lr] = losses
        print(f"lr={lr:<5} final loss={losses[-1]:.4f}  predictions={preds}")

    # loss curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(results[0.5], label="lr=0.5 (good)")
    ax[0].plot(results[0.01], label="lr=0.01 (too slow)")
    ax[0].set_title("XOR loss: good vs slow learning"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE"); ax[0].legend()
    ax[1].plot(results[5], color="red", label="lr=5 (too high)")
    ax[1].set_title("XOR loss: lr=5 (unstable / diverging)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    path = os.path.join(OUT, "task1_loss_curves.png")
    plt.savefig(path, dpi=120)
    print("\nLoss curves saved to", path)

    print("""
=== Short explanations ===
Dropout       : randomly zeroes a fraction of neurons in each training step so the network cannot rely on
                any single neuron. It acts like averaging many small networks and reduces overfitting (off at eval time).
BatchNorm     : normalises each layer's inputs to zero mean / unit variance per mini-batch, then rescales with
                learnable gamma and beta. This stabilises and speeds up training and allows higher learning rates.
Early stopping: monitor validation loss/accuracy each epoch and stop when it has not improved for N epochs (patience).
                Keep the best weights, so you stop before the model starts to overfit.
""")

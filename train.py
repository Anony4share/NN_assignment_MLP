# train_utils.py
import numpy as np
from model import MLP
import matplotlib.pyplot as plt

def train_model(
    X_train, y_train, X_val, y_val,
    input_size=3072,
    hidden_sizes=[128],  # Changed to a list to support multiple hidden layers
    output_size=10,
    model=None,
    lr=0.01,
    batch_size=32,
    epochs=100,
    optimizer="sgd",
    l2_lambda=0.0,
    verbose=False,
    save_best=False,
    save_path="best_model.npz"
):
    if model:
        model = model
    else:
        assert input_size and hidden_sizes and output_size
        model = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,  # Updated to pass the list of hidden sizes
            output_size=output_size,
            l2_lambda=l2_lambda
        )

    model._init_optimizer(optimizer)
    m = X_train.shape[0]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuff, y_shuff = X_train[indices], y_train[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuff[i:i + batch_size]
            y_batch = y_shuff[i:i + batch_size]

            A2, cache = model.forward(X_batch)
            grads = model.backward(cache, y_batch)
            model.update_params(grads, lr, optimizer)

        # 每个 epoch 评估
        A2_train, _ = model.forward(X_train)
        loss_train = model.cross_entropy_loss(A2_train, y_train)
        acc_train = np.mean(np.argmax(A2_train, axis=1) == y_train)

        A2_val, _ = model.forward(X_val)
        loss_val = model.cross_entropy_loss(A2_val, y_val)
        acc_val = np.mean(np.argmax(A2_val, axis=1) == y_val)

        history["train_loss"].append(loss_train)
        history["val_loss"].append(loss_val)
        history["train_acc"].append(acc_train)
        history["val_acc"].append(acc_val)

        if verbose:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {loss_train:.4f}, Acc: {acc_train:.4f} | "
                  f"Val Loss: {loss_val:.4f}, Acc: {acc_val:.4f}")

        # 保存最优模型（以 val_acc 为指标）
        if save_best and acc_val > best_val_acc:
            best_val_acc = acc_val
            best_weights = {}
            for i in range(1, len(model.weights) + 1):
                best_weights[f"W{i}"] = model.weights[i - 1].copy()
                best_weights[f"b{i}"] = model.biases[i - 1].copy()

    if save_best and best_weights:
        np.savez(save_path, **best_weights)
        if verbose:
            print(f"Best model saved to {save_path} with val_acc = {best_val_acc:.4f}")

    return model, history


def load_weights(model, path):
    weights = np.load(path)
    for i in range(1, len(model.weights) + 1):
        model.weights[i - 1] = weights[f"W{i}"]
        model.biases[i - 1] = weights[f"b{i}"]

def plot_training_history(history):
    """
    绘制训练和验证的损失与准确率曲线。

    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_training_plot(history, save_path="training_history.png"):

    epochs = range(1, len(history["train_loss"]) + 1)

    # Create the plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
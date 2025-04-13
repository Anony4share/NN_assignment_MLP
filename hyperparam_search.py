import numpy as np
import os
from model import MLP
from train import train_model, save_training_plot
from dataset import load_cifar10_data

def hyperparam_search():
    # Load CIFAR-10 dataset
    train_data, train_labels, test_data, test_labels = load_cifar10_data("./sample_data/cifar-10-data")

    # Split training data into training and validation sets
    val_split = 0.1
    num_val_samples = int(len(train_data) * val_split)
    X_val, y_val = train_data[:num_val_samples], train_labels[:num_val_samples]
    X_train, y_train = train_data[num_val_samples:], train_labels[num_val_samples:]

    # Define hyperparameter grid
    hidden_sizes_list = [[128], [256], [128, 256], [128, 128], [256, 512]]
    learning_rates = [0.001, 0.01]
    batch_sizes = [32, 64]
    epochs = 30
    optimizer = "adam"
    l2_lambdas = [0.0, 0.001]

    best_val_acc = 0.0
    best_hyperparams = None

    # Perform grid search
    for hidden_sizes in hidden_sizes_list:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for l2_lambda in l2_lambdas:
                    print(f"Training with hidden_sizes={hidden_sizes}, lr={lr}, batch_size={batch_size}, l2_lambda={l2_lambda}")

                    # Generate directory name based on hyperparameters
                    dir_name = f"hidden_{hidden_sizes}_lr_{lr}_batch_{batch_size}_l2_{l2_lambda}"
                    os.makedirs(dir_name, exist_ok=True)

                    # Initialize model
                    model = MLP(input_size=3072, hidden_sizes=hidden_sizes, output_size=10, l2_lambda=l2_lambda)

                    # Train model
                    _, history = train_model(
                        X_train, y_train, X_val, y_val,
                        model=model,
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        optimizer=optimizer,
                        l2_lambda=l2_lambda,
                        verbose=False,
                        save_best=True,  # Enable saving the best weights
                        save_path=os.path.join(dir_name, "best_weights.npz")  # Save directly to the directory
                    )

                    # Get the highest validation accuracy
                    val_acc = max(history["val_acc"])
                    print(f"Validation Accuracy: {val_acc:.4f}")

                    # Evaluate on the test set
                    A2_test, _ = model.forward(test_data)
                    test_accuracy = np.mean(np.argmax(A2_test, axis=1) == test_labels)
                    print(f"Test Accuracy: {test_accuracy:.4f}")

                    # Save training history and plot if this is the best model so far
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_hyperparams = (hidden_sizes, lr, batch_size, l2_lambda)

                    # Rename directory to include validation accuracy
                    new_dir_name = f"{dir_name}_valacc_{val_acc:.4f}_testacc_{test_accuracy:.4f}"
                    os.rename(dir_name, new_dir_name)

                    # Save training history
                    np.savez(os.path.join(new_dir_name, "history.npz"), **history)

                    # Save training plot
                    save_training_plot(history, save_path=os.path.join(new_dir_name, "training_history.png"))

    print(f"Best Hyperparameters: hidden_sizes={best_hyperparams[0]}, lr={best_hyperparams[1]}, batch_size={best_hyperparams[2]}, l2_lambda={best_hyperparams[3]}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    hyperparam_search()

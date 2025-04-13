import numpy as np

class MLP():

    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return x > 0

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止溢出
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def add_hidden_layer(self, input_size, output_size):
        W = np.random.randn(input_size, output_size) * 0.01
        b = np.zeros((1, output_size))
        return W, b

    def __init__(self, input_size, hidden_sizes, output_size, l2_lambda=0, seed=42):
        np.random.seed(seed)

        self.hidden_layers = len(hidden_sizes)
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        prev_size = input_size
        for size in hidden_sizes:
            W, b = self.add_hidden_layer(prev_size, size)
            self.weights.append(W)
            self.biases.append(b)
            prev_size = size

        # Output layer
        W, b = self.add_hidden_layer(prev_size, output_size)
        self.weights.append(W)
        self.biases.append(b)

        self.l2_lambda = l2_lambda
        self.opt_state = {}

    def forward(self, X):
        cache = [X]
        A = X

        # Forward pass through hidden layers
        for i in range(self.hidden_layers):
            Z = A.dot(self.weights[i]) + self.biases[i]
            A = MLP.relu(Z)
            cache.extend([Z, A])

        # Forward pass through output layer
        Z = A.dot(self.weights[-1]) + self.biases[-1]
        A = MLP.softmax(Z)
        cache.extend([Z, A])

        return A, cache

    def backward(self, cache, y):
        m = y.shape[0]
        grads = {}

        # Unpack cache
        X = cache[0]
        A_prev = cache[-3]
        A_L = cache[-1]

        # Backprop through output layer
        dZ = A_L.copy()
        dZ[np.arange(m), y] -= 1
        dZ /= m

        grads[f"dW{self.hidden_layers + 1}"] = A_prev.T @ dZ
        grads[f"db{self.hidden_layers + 1}"] = np.sum(dZ, axis=0, keepdims=True)

        dA = dZ @ self.weights[-1].T

        # Backprop through hidden layers
        for i in range(self.hidden_layers, 0, -1):
            Z = cache[2 * i - 1]
            A_prev = cache[2 * i - 2]

            dZ = dA * MLP.relu_derivative(Z)
            grads[f"dW{i}"] = A_prev.T @ dZ
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)

            dA = dZ @ self.weights[i - 1].T

        return grads

    def _params(self):
        params = {}
        for i in range(1, self.hidden_layers + 2):
            params[f"W{i}"] = self.weights[i - 1]
            params[f"b{i}"] = self.biases[i - 1]
        return params

    # def _params(self):
    #     return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def predict(self, X):
        A2, _ = self.forward(X)
        return np.argmax(A2, axis=1)

    def cross_entropy_loss(self, A2, y):
        m = y.shape[0]
        log_likelihood = -np.log(A2[range(m), y] + 1e-9)
        loss = np.mean(log_likelihood)

        # L2 正则化
        if self.l2_lambda > 0:
            l2_term = 0.5 * self.l2_lambda * sum(np.sum(W**2) for W in self.weights)
            loss += l2_term / m

        return loss

    def _init_optimizer(self, optimizer):
        if optimizer == "momentum":
            self.opt_state = {k: np.zeros_like(v) for k, v in self._params().items()}
        elif optimizer == "adam":
            self.opt_state = {
                "m": {k: np.zeros_like(v) for k, v in self._params().items()},
                "v": {k: np.zeros_like(v) for k, v in self._params().items()},
                "t": 0
            }

    def update_params(self, grads, lr, optimizer="sgd", beta1=0.9, beta2=0.999, eps=1e-8):
        params = self._params()

        if optimizer == "sgd":
            for k in grads:
                param_name = k[1:]  # 'dW1' -> 'W1'
                params[param_name][...] -= lr * grads[k]

        elif optimizer == "momentum":
            for k in grads:
                param_name = k[1:]
                self.opt_state[param_name] = 0.9 * self.opt_state[param_name] + lr * grads[k]
                params[param_name][...] -= self.opt_state[param_name]

        elif optimizer == "adam":
            self.opt_state["t"] += 1
            for k in grads:
                param_name = k[1:]
                m = self.opt_state["m"][param_name]
                v = self.opt_state["v"][param_name]
                g = grads[k]

                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * (g ** 2)

                m_hat = m / (1 - beta1 ** self.opt_state["t"])
                v_hat = v / (1 - beta2 ** self.opt_state["t"])

                params[param_name][...] -= lr * m_hat / (np.sqrt(v_hat) + eps)


"""MLP escrito do zero — núcleo reutilizado, sem alterações, nos Exercises 2, 3 e 4.

Camadas ocultas com ativação tanh; saída com sigmoide (``task="binary"``, perda
BCE) ou softmax (``task="multiclass"``, perda entropia cruzada categórica).
Gradiente descendente em lote completo (todo o treino a cada época).
"""

import numpy as np


class MLP:
    def __init__(self, layer_sizes: list[int], task: str, rng: np.random.Generator, eta: float):
        """``layer_sizes`` inclui entrada e saída, ex. [2, 8, 1] ou [4, 16, 16, 3]."""
        assert task in ("binary", "multiclass")
        self.layer_sizes = layer_sizes
        self.task = task
        self.eta = eta
        self.weights = [
            rng.normal(0, 1 / np.sqrt(layer_sizes[i]), size=(layer_sizes[i + 1], layer_sizes[i]))
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    # --- ativações ---------------------------------------------------------
    @staticmethod
    def _hidden(z):
        return np.tanh(z)

    @staticmethod
    def _hidden_grad(a):
        return 1 - a**2

    def _output(self, z):
        if self.task == "binary":
            return 1 / (1 + np.exp(-z))
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    # --- passagem direta -----------------------------------------------------
    def forward(self, X: np.ndarray):
        activations = [X]
        a = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self._hidden(a @ w.T + b)
            activations.append(a)
        y_hat = self._output(a @ self.weights[-1].T + self.biases[-1])
        activations.append(y_hat)
        return activations

    def loss(self, y_hat: np.ndarray, y_target: np.ndarray) -> float:
        eps = 1e-12
        if self.task == "binary":
            y_target = y_target.reshape(-1, 1)
            return float(-np.mean(y_target * np.log(y_hat + eps) + (1 - y_target) * np.log(1 - y_hat + eps)))
        return float(-np.mean(np.sum(y_target * np.log(y_hat + eps), axis=-1)))

    # --- passagem reversa ----------------------------------------------------
    def backward(self, activations: list[np.ndarray], y_target: np.ndarray):
        y_hat = activations[-1]
        n = y_target.shape[0]
        y_target = y_target.reshape(-1, 1) if self.task == "binary" else y_target

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # delta da saída: sigmoide+BCE e softmax+CCE dão a mesma forma, (y_hat - y)/n
        delta = (y_hat - y_target) / n
        grads_w[-1] = delta.T @ activations[-2]
        grads_b[-1] = delta.sum(axis=0)

        for layer in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[layer + 1]) * self._hidden_grad(activations[layer + 1])
            grads_w[layer] = delta.T @ activations[layer]
            grads_b[layer] = delta.sum(axis=0)

        return grads_w, grads_b

    def update(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.eta * grads_w[i]
            self.biases[i] -= self.eta * grads_b[i]

    # --- laço de treino ------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> list[float]:
        if self.task == "multiclass" and y.ndim == 1:
            n_classes = self.layer_sizes[-1]
            y = np.eye(n_classes)[y]

        losses = []
        for _ in range(epochs):
            activations = self.forward(X)
            losses.append(self.loss(activations[-1], y))
            grads_w, grads_b = self.backward(activations, y)
            self.update(grads_w, grads_b)
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat = self.forward(X)[-1]
        if self.task == "binary":
            return (y_hat.ravel() >= 0.5).astype(int)
        return y_hat.argmax(axis=1)

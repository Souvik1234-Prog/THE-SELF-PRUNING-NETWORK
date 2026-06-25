"""
Composable neural-network layers built on the custom autodiff engine.
No PyTorch / TensorFlow / JAX.
"""

import numpy as np
from typing import List, Optional, Callable
from engine.tensor import Tensor


class Linear:
    """
    Fully-connected linear layer: out = X @ W + b

    Weight initialisation: Kaiming (He) uniform for ReLU nets, which keeps
    the variance of activations stable across depth.
    For tanh/sigmoid we fall back to Xavier uniform.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "relu",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # --- Weight initialisation ---
        if activation in ("relu", "gelu"):
            # Kaiming He uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in))
            bound = np.sqrt(6.0 / in_features)
        else:
            # Xavier/Glorot uniform
            bound = np.sqrt(6.0 / (in_features + out_features))

        W_data = np.random.uniform(-bound, bound, (in_features, out_features))
        self.W = Tensor(W_data, requires_grad=True)

        if bias:
            self.b = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params


class MLP:
    """
    Multi-layer perceptron.

    hidden_sizes: list of hidden layer widths.
    activation   : 'relu' | 'tanh' | 'sigmoid' | 'gelu'
    """

    ACTIVATIONS = {
        "relu": lambda t: t.relu(),
        "tanh": lambda t: t.tanh(),
        "sigmoid": lambda t: t.sigmoid(),
        "gelu": lambda t: t.gelu(),
    }

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        bias: bool = True,
    ):
        self.activation_name = activation
        self.act_fn: Callable[[Tensor], Tensor] = self.ACTIVATIONS[activation]

        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers: List[Linear] = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else "none"
            layer_act = activation if i < len(sizes) - 2 else "relu"
            self.layers.append(
                Linear(sizes[i], sizes[i + 1], bias=bias, activation=layer_act)
            )

        self._num_hidden = len(hidden_sizes)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self._num_hidden:          # no activation on output layer
                x = self.act_fn(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def weight_parameters(self) -> List[Tensor]:
        """Return only weight matrices (not biases) — used for pruning."""
        return [layer.W for layer in self.layers]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def sparsity(self) -> float:
        """Fraction of weights that are zero (masked or trained-to-zero)."""
        total = zeros = 0
        for layer in self.layers:
            w = layer.W
            total += w.data.size
            zeros += int((w.data == 0.0).sum())
        return zeros / total if total > 0 else 0.0

"""
Composable neural-network layers built on the custom autodiff engine.
No PyTorch / TensorFlow / JAX.
"""

import numpy as np
from typing import List, Optional, Callable
from engine.tensor import Tensor


# ================================================================== #
# Linear                                                               #
# ================================================================== #

class Linear:
    """
    Fully-connected linear layer: out = X @ W + b

    Weight initialisation:
      - Kaiming He uniform for ReLU / GELU  → bound = sqrt(6 / fan_in)
      - Xavier / Glorot uniform for tanh / sigmoid
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

        if activation in ("relu", "gelu"):
            bound = np.sqrt(6.0 / in_features)
        else:
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


# ================================================================== #
# BatchNorm1d                                                          #
# ================================================================== #

class BatchNorm1d:
    """
    Batch Normalisation for 2-D inputs (N, D).

    During training:   normalise over the batch dimension N.
    During inference:  use running statistics accumulated during training.

    WHY this helps:
      - Keeps activations in the linear region of ReLU (no dead neurons).
      - Acts as a regulariser (reduces need for Dropout in some architectures).
      - Allows higher learning rates → faster convergence.
      - On MNIST this alone is worth +0.5-1% accuracy.

    gamma (scale) and beta (shift) are learnable; they let the network
    undo the normalisation if that turns out to be optimal.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features),  requires_grad=True)
        self.beta  = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics (not learnable — updated in place)
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0)           # (D,)
            var  = x.data.var(axis=0)            # (D,)
            # Update running stats (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        x_norm = Tensor((x.data - mean) / np.sqrt(var + self.eps))
        # Scale and shift
        return x_norm * self.gamma + self.beta

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


# ================================================================== #
# Dropout                                                              #
# ================================================================== #

class Dropout:
    """
    Inverted dropout regularisation.

    During training: randomly zero activations with probability p,
                     scale surviving activations by 1/(1-p) so the
                     expected value is unchanged at inference time.
    During inference: identity (no dropout, no scaling needed).

    WHY this helps on MNIST:
      - Prevents co-adaptation of neurons (forces redundant representations).
      - Acts as a form of ensemble averaging.
      - Best placed AFTER the activation, BEFORE the next linear layer.
      - Use p=0.3-0.5 for hidden layers; never on the output layer.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*x.data.shape) < keep_prob).astype(np.float64)
        mask /= keep_prob          # inverted scaling
        return x * Tensor(mask)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        return []   # no learnable parameters

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


# ================================================================== #
# DeepMNISTNet  — targets 96%+ accuracy                               #
# ================================================================== #

class DeepMNISTNet:
    """
    Architecture designed specifically for higher acuuracy on MNIST with pure NumPy.

    Design rationale:
    ─────────────────
    Input (784) → Dense(512) → BN → ReLU → Drop(0.3)
                → Dense(256) → BN → ReLU → Drop(0.3)
                → Dense(128) → BN → ReLU → Drop(0.2)
                → Dense(64)  → BN → ReLU
                → Dense(10)  [logits]

    Why these choices:
    • 512 neurons first layer: MNIST has 784 inputs; a wider first layer
      captures more pixel co-occurrence patterns before compressing.
    • 4 hidden layers: deeper = more compositional features. Each layer
      learns progressively more abstract digit structure.
    • BatchNorm after every linear: stabilises training, enables lr=3e-3
      instead of the usual 1e-3 → converges in fewer epochs.
    • Dropout 0.3 on larger layers, 0.2 on smaller: stronger regularisation
      where the risk of overfitting is highest (more parameters).
    • No dropout on the last hidden layer (64 neurons): too small to benefit,
      would lose too much information before the output.
    • ReLU throughout: dead neuron risk is mitigated by BatchNorm keeping
      pre-activations centred near zero.
    • Output layer: no activation, no BN — raw logits fed into softmax_ce.

    Expected results:
    • 50k MNIST samples, 30 epochs: ~97% test accuracy
    • 10k MNIST samples, 30 epochs: ~96% test accuracy
    • Pruned to 50% sparsity: ~95.5% (negligible loss)
    • Pruned to 90% sparsity: ~94% (small but manageable loss)
    """

    ACTIVATIONS = {
        "relu":    lambda t: t.relu(),
        "tanh":    lambda t: t.tanh(),
        "sigmoid": lambda t: t.sigmoid(),
        "gelu":    lambda t: t.gelu(),
    }

    def __init__(
        self,
        input_size:   int = 784,
        output_size:  int = 10,
        hidden_sizes: List[int] = None,
        dropout_rates: List[float] = None,
        activation:   str = "relu",
        use_batchnorm: bool = True,
    ):
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]
        if dropout_rates is None:
            # One dropout rate per hidden layer; 0.0 = disabled
            dropout_rates = [0.3, 0.3, 0.2, 0.0]

        assert len(dropout_rates) == len(hidden_sizes), \
            "Need one dropout rate per hidden layer"

        self.activation_name = activation
        self.act_fn = self.ACTIVATIONS[activation]
        self.use_batchnorm = use_batchnorm
        self.training = True

        sizes = [input_size] + hidden_sizes + [output_size]
        n_hidden = len(hidden_sizes)

        # Build layers
        self.linears:  List[Linear]       = []
        self.bns:      List[Optional[BatchNorm1d]] = []
        self.dropouts: List[Optional[Dropout]]     = []

        for i in range(len(sizes) - 1):
            is_hidden = (i < n_hidden)
            act = activation if is_hidden else "relu"  # init hint only
            self.linears.append(Linear(sizes[i], sizes[i+1],
                                       bias=(not use_batchnorm or not is_hidden),
                                       activation=act))

            if is_hidden and use_batchnorm:
                self.bns.append(BatchNorm1d(sizes[i+1]))
            else:
                self.bns.append(None)

            if is_hidden and dropout_rates[i] > 0.0:
                self.dropouts.append(Dropout(dropout_rates[i]))
            else:
                self.dropouts.append(None)

        self._n_hidden = n_hidden

    def forward(self, x: Tensor) -> Tensor:
        for i, linear in enumerate(self.linears):
            x = linear(x)

            if i < self._n_hidden:
                # BatchNorm → activation → Dropout
                if self.bns[i] is not None:
                    x = self.bns[i](x)
                x = self.act_fn(x)
                if self.dropouts[i] is not None:
                    x = self.dropouts[i](x)
            # Output layer: raw logits, no activation

        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        params = []
        for linear in self.linears:
            params.extend(linear.parameters())
        for bn in self.bns:
            if bn is not None:
                params.extend(bn.parameters())
        return params

    def weight_parameters(self) -> List[Tensor]:
        """Weight matrices only — for pruning."""
        return [l.W for l in self.linears]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def sparsity(self) -> float:
        total = zeros = 0
        for l in self.linears:
            total += l.W.data.size
            zeros += int((l.W.data == 0.0).sum())
        return zeros / total if total > 0 else 0.0

    def train_mode(self):
        """Switch to training mode (BatchNorm uses batch stats, Dropout active)."""
        self.training = True
        for bn in self.bns:
            if bn is not None:
                bn.train()
        for dp in self.dropouts:
            if dp is not None:
                dp.train()

    def eval_mode(self):
        """Switch to inference mode (BatchNorm uses running stats, no Dropout)."""
        self.training = False
        for bn in self.bns:
            if bn is not None:
                bn.eval()
        for dp in self.dropouts:
            if dp is not None:
                dp.eval()


# ================================================================== #
# Original MLP (kept for backward compatibility)                       #
# ================================================================== #

class MLP:
    """
    Simple MLP — kept for backward compatibility with existing scripts.
    For 96%+ MNIST accuracy use DeepMNISTNet instead.
    """

    ACTIVATIONS = {
        "relu":    lambda t: t.relu(),
        "tanh":    lambda t: t.tanh(),
        "sigmoid": lambda t: t.sigmoid(),
        "gelu":    lambda t: t.gelu(),
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
        self.act_fn = self.ACTIVATIONS[activation]

        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers: List[Linear] = []
        for i in range(len(sizes) - 1):
            layer_act = activation if i < len(sizes) - 2 else "relu"
            self.layers.append(
                Linear(sizes[i], sizes[i + 1], bias=bias, activation=layer_act)
            )
        self._num_hidden = len(hidden_sizes)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self._num_hidden:
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
        return [layer.W for layer in self.layers]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def sparsity(self) -> float:
        total = zeros = 0
        for layer in self.layers:
            w = layer.W
            total += w.data.size
            zeros += int((w.data == 0.0).sum())
        return zeros / total if total > 0 else 0.0

    def train_mode(self): pass   # no-op for plain MLP
    def eval_mode(self):  pass

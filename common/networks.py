"""Small network building blocks shared across algorithms.

Keep this dumb: just an MLP factory. Anything algorithm-specific (twin critics,
squashed Gaussian heads, etc.) lives in the algorithm file, not here.
"""

import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Build a multi-layer perceptron from a list of layer sizes.

    sizes=[obs_dim, 256, 256, act_dim] -> Linear/ReLU stack with the given
    output activation on the final layer.
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

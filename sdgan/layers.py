"""Small reusable neural network building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class LipSwish(nn.Module):
    """Lipschitz-friendly activation using a scaled SiLU (`torch.nn.functional.silu`)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies LipSwish to the input.

        Args:
            x: Arbitrary-shaped tensor.

        Returns:
            Scaled SiLU activation of ``x``.
        """
        return 0.909 * torch.nn.functional.silu(x)


class MLP(nn.Module):
    """Multi-layer perceptron with LipSwish activations."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        mlp_size: int,
        num_layers: int,
        tanh: bool,
    ) -> None:
        """Builds stacked linear layers.

        Args:
            in_size: Input feature dimension.
            out_size: Output feature dimension.
            mlp_size: Hidden width for all but the final layer.
            num_layers: Number of linear layers (including output).
            tanh: If True, applies ``TanH`` after the last linear layer.
        """
        super().__init__()
        layers_list: list[nn.Module] = [
            nn.Linear(in_size, mlp_size),
            LipSwish(),
        ]
        for _ in range(num_layers - 1):
            layers_list.append(nn.Linear(mlp_size, mlp_size))
            layers_list.append(LipSwish())
        layers_list.append(nn.Linear(mlp_size, out_size))
        if tanh:
            layers_list.append(nn.Tanh())
        self._model = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the MLP forward pass.

        Args:
            x: Input tensor shaped ``[..., in_size]``.

        Returns:
            Output tensor shaped ``[..., out_size]``.
        """
        return self._model(x)

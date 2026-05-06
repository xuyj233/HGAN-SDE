"""SDE-backed generator producing neural CDE interpolation coefficients."""

from __future__ import annotations

import torch
import torchcde
import torchsde

from sdgan.layers import MLP


class GeneratorFunc(torch.nn.Module):
    """Stratonovich SDE drift and diffusion nets for the latent state."""

    sde_type = "stratonovich"
    noise_type = "general"

    def __init__(
        self,
        noise_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
    ) -> None:
        """Initializes drift and diffusion MLPs.

        Args:
            noise_size: Width of the noise matrix.
            hidden_size: Latent state dimension.
            mlp_size: Hidden size for internal MLPs.
            num_layers: Number of linear layers in each MLP.
        """
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(
            1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True
        )

    def f_and_g(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes drift ``f`` and diffusion ``g`` at time ``t``.

        Args:
            t: Scalar time tensor broadcastable to batch.
            x: Latent state ``(batch, hidden_size)``.

        Returns:
            Drift vector ``(batch, hidden)`` and diffusion tensor
            ``(batch, hidden, noise_size)``.
        """
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx), self._diffusion(tx).view(
            x.size(0), self._hidden_size, self._noise_size
        )


class Generator(torch.nn.Module):
    """Maps IID noise through an SDE and reads out observation-space paths."""

    def __init__(
        self,
        data_size: int,
        initial_noise_size: int,
        noise_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
    ) -> None:
        """Builds encoder from noise to ``x0``, SDE function, and readout linear map.

        Args:
            data_size: Dimension of modeled observations (excluding time channel).
            initial_noise_size: Dimension of Gaussian noise injected at ``t0``.
            noise_size: SDE diffusion noise dimension.
            hidden_size: Latent dimension integrated by ``torchsde``.
            mlp_size: Width of auxiliary MLPs.
            num_layers: Depth hyperparameter forwarded to ``MLP``.
        """
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, ts: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Samples trajectories on ``ts`` and returns linear spline coefficients.

        Args:
            ts: Increasing time grid ``(T,)`` on the computation device.
            batch_size: Number of independent sampled paths.

        Returns:
            Tensor ``(batch_size, T, 1 + data_size)`` of coefficients compatible with
            ``torchcde.LinearInterpolation``.
        """
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)
        xs = torchsde.sdeint_adjoint(
            self._func,
            x0,
            ts,
            method="reversible_heun",
            dt=1,
            adjoint_method="adjoint_reversible_heun",
        )
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)
        ts_b = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts_b, ys], dim=2))

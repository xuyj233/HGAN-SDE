"""Conditional discriminator variants built from Hermite features or neural CDEs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchcde

from sdgan.layers import MLP
from sdgan.math_utils import hermite_function, hermite_poly, standard_normal_density


class DiscriminatorFunc(torch.nn.Module):
    """Right-hand side for latent dynamics used by ``forward_type6``."""

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
    ) -> None:
        """Initializes a single fused MLP that produces hidden Jacobians."""
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size
        self._module = MLP(
            1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Evaluates the controlled vector field at ``(t, h)``.

        Args:
            t: Time tensor expanded per batch row.
            h: Hidden state ``(batch, hidden)``.

        Returns:
            Reshaped dynamics ``(batch, hidden, 1 + data_size)``.
        """
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    """Routes among multiple scoring heads operating on spline coefficients."""

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
        discriminator_type: str = "type1",
        num_terms: int = 3,
    ) -> None:
        """Constructs discriminator components and selects a forward delegate.

        Args:
            data_size: Number of feature channels excluding time.
            hidden_size: CDE latent width for ``type6``.
            mlp_size: Hidden size for constituent MLPs.
            num_layers: Depth forwarded to nested ``MLP`` modules.
            discriminator_type: One of ``{"type1", ..., "type6"}``.
            num_terms: Number of Hermite basis terms used in Hermite-heavy variants.

        Raises:
            ValueError: If ``discriminator_type`` is unknown.
        """
        super().__init__()
        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)
        self.num_terms = num_terms
        self.hidden = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(64, self.num_terms)

        type_map = {
            "type1": self.forward_type1,
            "type2": self.forward_type2,
            "type3": self.forward_type3,
            "type4": self.forward_type4,
            "type5": self.forward_type5,
            "type6": self.forward_type6,
        }
        if discriminator_type not in type_map:
            raise ValueError(f"Unknown discriminator_type: {discriminator_type}")
        self.forward_fn = type_map[discriminator_type]

    def forward_type1(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """Softmax surrogate likelihood using physicists' Hermite polynomials."""
        batch_size, seq_length, _ = ys_coeffs.size()
        timestamps = ys_coeffs[:, :, 0]
        values = ys_coeffs[:, :, 1]
        x0 = values[:, :-1]
        xt = values[:, 1:]
        t = timestamps[:, 1:]
        xt_t_x0 = torch.stack([xt, t, x0], dim=-1).view(-1, 3)
        c_n = self.output_layer(self.hidden(xt_t_x0))
        xt_flat = xt.contiguous().view(-1)
        xt_flat_numpy = xt_flat.detach().cpu().numpy()
        h_numpy = np.stack(
            [hermite_poly(n)(xt_flat_numpy) for n in range(self.num_terms)], axis=-1
        )
        h_tensor = torch.tensor(h_numpy, dtype=torch.float32, device=ys_coeffs.device)
        p_xt_x0 = (c_n * h_tensor).sum(dim=-1)
        p_xt_x0 = p_xt_x0.view(batch_size, seq_length - 1)
        p_xt_x0 = nn.functional.softmax(p_xt_x0, dim=0)
        log_likelihood = torch.log(torch.clamp(p_xt_x0, min=1e-10)).sum(dim=-1)
        return log_likelihood.mean()

    def forward_type2(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """Bernoulli transition likelihood with Hermite features."""
        batch_size, seq_length, _ = ys_coeffs.size()
        timestamps = ys_coeffs[:, :, 0]
        values = ys_coeffs[:, :, 1]
        x_prev = values[:, :-1]
        x_curr = values[:, 1:]
        delta = timestamps[:, 1:]
        inputs = torch.stack([x_curr, delta, x_prev], dim=-1).view(-1, 3)
        c_n = self.output_layer(self.hidden(inputs))
        x_curr_flat = x_curr.contiguous().view(-1)
        x_curr_flat_np = x_curr_flat.detach().cpu().numpy()
        h_np = np.stack([hermite_poly(n)(x_curr_flat_np) for n in range(self.num_terms)], axis=-1)
        h_tensor = torch.tensor(h_np, dtype=torch.float32, device=ys_coeffs.device)
        raw_scores = (c_n * h_tensor).sum(dim=-1)
        raw_scores = raw_scores.view(batch_size, seq_length - 1)
        p_xt_xprev = torch.sigmoid(raw_scores)
        log_likelihood = torch.log(torch.clamp(p_xt_xprev, min=1e-10)).sum(dim=-1)
        return log_likelihood.mean()

    def forward_type3(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """Endpoint-conditioned Bernoulli score with Hermite basis."""
        timestamps = ys_coeffs[:, :, 0]
        values = ys_coeffs[:, :, 1]
        x_0 = values[:, 0]
        x_t = values[:, -1]
        delta = timestamps[:, -1] - timestamps[:, 0]
        inputs = torch.stack([x_t, x_0, delta], dim=-1)
        c_n = self.output_layer(self.hidden(inputs))
        x_t_np = x_t.detach().cpu().numpy()
        h_np = np.stack([hermite_poly(n)(x_t_np) for n in range(self.num_terms)], axis=-1)
        h_tensor = torch.tensor(h_np, dtype=torch.float32, device=ys_coeffs.device)
        p_xfinal_x0 = (c_n * h_tensor).sum(dim=-1)
        p_xfinal_x0 = torch.sigmoid(p_xfinal_x0)
        return p_xfinal_x0.mean()

    def forward_type4(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """Transition density proxy using orthonormal Hermite functions."""
        batch_size, seq_length, _ = ys_coeffs.size()
        timestamps = ys_coeffs[:, :, 0]
        values = ys_coeffs[:, :, 1]
        x_prev = values[:, :-1]
        x_curr = values[:, 1:]
        delta = timestamps[:, 1:]
        inputs = torch.stack([x_curr, delta, x_prev], dim=-1).view(-1, 3)
        c_n = self.output_layer(self.hidden(inputs))
        x_curr_flat = x_curr.contiguous().view(-1)
        x_curr_flat_np = x_curr_flat.detach().cpu().numpy()
        h_np = np.stack(
            [hermite_function(n, x_curr_flat_np) for n in range(self.num_terms)], axis=-1
        )
        h_tensor = torch.tensor(h_np, dtype=torch.float32, device=ys_coeffs.device)
        raw_scores = (c_n * h_tensor).sum(dim=-1)
        raw_scores = raw_scores.view(batch_size, seq_length - 1)
        p_xt_xprev = torch.clamp(raw_scores, min=1e-10)
        log_likelihood = torch.log(torch.clamp(p_xt_xprev, min=1e-10)).sum(dim=-1)
        return log_likelihood.mean()

    def forward_type5(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """Like ``forward_type4`` but multiplies a sigmoid score by a Gaussian density."""
        batch_size, seq_length, _ = ys_coeffs.size()
        timestamps = ys_coeffs[:, :, 0]
        values = ys_coeffs[:, :, 1]
        x_prev = values[:, :-1]
        x_curr = values[:, 1:]
        delta = timestamps[:, 1:]
        inputs = torch.stack([x_curr, delta, x_prev], dim=-1).view(-1, 3)
        c_n = self.output_layer(self.hidden(inputs))
        x_curr_flat = x_curr.contiguous().view(-1)
        x_curr_flat_np = x_curr_flat.detach().cpu().numpy()
        h_np = np.stack(
            [hermite_function(n, x_curr_flat_np) for n in range(self.num_terms)], axis=-1
        )
        h_tensor = torch.tensor(h_np, dtype=torch.float32, device=ys_coeffs.device)
        raw_scores = (c_n * h_tensor).sum(dim=-1)
        raw_scores = raw_scores.view(batch_size, seq_length - 1)
        phi_xt_np = standard_normal_density(x_curr_flat_np)
        phi_xt = torch.tensor(phi_xt_np, dtype=torch.float32, device=ys_coeffs.device)
        phi_xt = phi_xt.view(batch_size, seq_length - 1)
        p_xt_xprev = torch.sigmoid(raw_scores) * phi_xt
        log_likelihood = torch.log(torch.clamp(p_xt_xprev, min=1e-10)).sum(dim=-1)
        return log_likelihood.mean()

    def forward_type6(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """Neural CDE discriminator averaging a terminal linear readout."""
        y = torchcde.LinearInterpolation(ys_coeffs)
        y0 = y.evaluate(y.interval[0])
        h0 = self._initial(y0)
        hs = torchcde.cdeint(
            y,
            self._func,
            h0,
            y.interval,
            method="reversible_heun",
            backend="torchsde",
            dt=1.0,
            adjoint_method="adjoint_reversible_heun",
            adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()),
        )
        score = self._readout(hs[:, -1])
        return score.mean()

    def forward(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """Delegates to the implementation selected via ``discriminator_type``."""
        return self.forward_fn(ys_coeffs)

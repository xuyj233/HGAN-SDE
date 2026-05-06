"""Feature preprocessing for trajectory tensors."""

from __future__ import annotations

import torch


def normalize_features_by_initial_value(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes feature channels using statistics at initial time ``t = 0``.

    The tensor is assumed to have shape ``(batch, time_steps, features + 1)`` where the
    first channel indexes time and is copied through unchanged.

    Args:
        tensor: Input batch of augmented paths ``(B, T, 1 + F)``.

    Returns:
        Same shape as ``tensor``, with channels ``1..F`` affine-normalized using the
        mean and std at the first timestep. NaNs are masked when computing statistics
        and restored in the normalized output.
    """
    timestamps = tensor[:, :, 0]
    features = tensor[:, :, 1:]
    nan_mask = torch.isnan(features)
    y0 = features[:, 0, :].unsqueeze(1)
    y0_mean = y0.masked_fill(nan_mask[:, 0, :].unsqueeze(1), 0).mean(
        dim=0, keepdim=True
    )
    y0_std = y0.masked_fill(nan_mask[:, 0, :].unsqueeze(1), 0).std(
        dim=0, keepdim=True
    )
    y0_std = y0_std.masked_fill(y0_std == 0, 1)
    normalized_features = (features - y0_mean) / y0_std
    normalized_features[nan_mask] = float("nan")
    return torch.cat([timestamps.unsqueeze(-1), normalized_features], dim=-1)

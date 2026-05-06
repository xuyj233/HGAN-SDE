"""Training-time diagnostics and distribution metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torchcde

from sdgan.preprocess import normalize_features_by_initial_value


def evaluate_loss(
    ts: torch.Tensor,
    batch_size: int,
    dataloader: torch.utils.data.DataLoader,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    step: int,
) -> tuple[float, float, float]:
    """Computes surrogate adversarial loss without gradients.

    Args:
        ts: Simulation grid shared by generator samples.
        batch_size: Loader batch dimension used for averaging bookkeeping.
        dataloader: Supplies real spline coefficients.
        generator: Callable ``(ts, batch_size) -> coeffs``.
        discriminator: Callable mapping coeffs to scalar scores.
        step: Unused index retained for logging symmetry with callers.

    Returns:
        Tuple ``(avg_loss, avg_real_score, avg_generated_score)`` aggregated across the
        loader with simple sample counting.

    Raises:
        Nothing explicitly; assumes compatible tensor devices.
    """
    del step  # Maintained for API symmetry with training loop call sites.
    gen_device = next(generator.parameters()).device

    with torch.no_grad():
        total_samples = 0
        total_loss = 0.0
        total_real_score = 0.0
        total_generated_score = 0.0

        for real_samples, in dataloader:
            real_samples = real_samples.to(gen_device)
            gen_samples = generator(ts, batch_size)

            gen_score = discriminator(normalize_features_by_initial_value(gen_samples))
            real_score = discriminator(real_samples)

            total_generated_score += gen_score.mean().item() * batch_size
            total_real_score += real_score.mean().item() * batch_size

            loss = gen_score - real_score
            total_loss += loss.mean().item() * batch_size
            total_samples += batch_size

        avg_real_score = total_real_score / total_samples
        avg_gen_score = total_generated_score / total_samples

        print(
            f"Avg Real Score: {avg_real_score:.6f}, "
            f"Avg Generated Score: {avg_gen_score:.6f}"
        )
    return total_loss / total_samples, avg_real_score, avg_gen_score


def evaluate_metrics(
    ts: torch.Tensor,
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    num_samples: int = 10000,
    num_bins: int = 500,
    tail_percentiles: Iterable[float] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compares marginal distributions at the terminal time.

    Args:
        ts: Time grid for reconstructing paths from coefficients.
        generator: Maps noise to generated coefficients.
        dataloader: Source of real coefficients (first batch is consumed).
        device: Device for generator evaluation.
        num_samples: Truncation of the real batch for efficiency.
        num_bins: Histogram resolution for the MISE proxy.
        tail_percentiles: Lower-tail masses used for left/right quantile deltas.

    Returns:
        Dictionary with key ``"MISE"`` and nested ``"TailDiff"`` mapping descriptive
        labels to numeric gaps between generated and empirical quantiles.
    """
    if tail_percentiles is None:
        tail_percentiles = [0.01, 0.05]

    real_samples, = next(iter(dataloader))
    real_samples = real_samples.to(device)[:num_samples]
    batch_size = real_samples.size(0)

    with torch.no_grad():
        gen_samples = generator(ts, batch_size)

    real_eval = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_eval = real_eval[..., 1].cpu().numpy()

    gen_eval = torchcde.LinearInterpolation(gen_samples).evaluate(ts)
    gen_eval = gen_eval[..., 1].cpu().numpy()

    seq_len = real_eval.shape[1]
    t_idx = seq_len - 1

    real_t = real_eval[:, t_idx]
    gen_t = gen_eval[:, t_idx]

    data_min = min(real_t.min(), gen_t.min())
    data_max = max(real_t.max(), gen_t.max())
    if data_min == data_max:
        data_min -= 1e-8
        data_max += 1e-8

    bins = np.linspace(data_min, data_max, num_bins + 1)
    hist_r, _ = np.histogram(real_t, bins=bins, density=True)
    hist_g, _ = np.histogram(gen_t, bins=bins, density=True)

    bin_width = (data_max - data_min) / num_bins
    sq_diff = ((hist_r - hist_g) ** 2).sum() * bin_width
    mise = float(sq_diff)

    real_sorted = np.sort(real_t)
    gen_sorted = np.sort(gen_t)

    tail_results: dict[str, float] = {}
    for p in tail_percentiles:
        real_left = np.percentile(real_sorted, 100 * p)
        gen_left = np.percentile(gen_sorted, 100 * p)
        tail_results[f"left_{int(100 * p)}%"] = float(gen_left - real_left)

        real_right = np.percentile(real_sorted, 100 * (1 - p))
        gen_right = np.percentile(gen_sorted, 100 * (1 - p))
        tail_results[f"right_{int(100 * p)}%"] = float(gen_right - real_right)

    return {"MISE": mise, "TailDiff": tail_results}

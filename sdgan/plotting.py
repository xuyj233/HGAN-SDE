"""Visualization helpers for qualitative GAN diagnostics."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
import torchcde


def plot(
    ts: torch.Tensor,
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_plot_samples: int,
    plot_locs: list[float],
    save_dir: str,
) -> None:
    """Saves histograms and overlaid temporal curves for real vs generated paths.

    Args:
        ts: Time grid used to decode spline coefficients.
        generator: Callable ``(ts, batch)`` returning generated coefficients.
        dataloader: Iterable of real coefficient batches.
        num_plot_samples: Number of trajectories overlaid on the line plot.
        plot_locs: Fractional indices (in ``[0, 1]``) selecting histogram snapshots.
        save_dir: Output directory created if missing.

    Raises:
        AssertionError: If ``num_plot_samples`` exceeds the available batch size.
    """
    os.makedirs(save_dir, exist_ok=True)

    real_samples, = next(iter(dataloader))
    assert num_plot_samples <= real_samples.size(0)

    real_eval = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_eval = real_eval[..., 1]

    with torch.no_grad():
        gen_samples = generator(ts, real_eval.size(0)).cpu()
    gen_eval = torchcde.LinearInterpolation(gen_samples).evaluate(ts)
    gen_eval = gen_eval[..., 1]

    for prop in plot_locs:
        time_idx = int(prop * (real_eval.size(1) - 1))
        real_t = real_eval[:, time_idx].cpu().numpy()
        gen_t = gen_eval[:, time_idx].cpu().numpy()

        plt.hist(real_t, bins=32, alpha=0.7, label="Real", color="dodgerblue", density=True)
        range_width = gen_t.max() - gen_t.min()
        bin_width = (real_t.max() - real_t.min()) / 32 if len(real_t) > 1 else 1.0
        num_bins = max(1, int(range_width / (bin_width if bin_width > 1e-12 else 1.0)))
        plt.hist(gen_t, bins=num_bins, alpha=0.7, label="Generated", color="crimson", density=True)

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Distribution at time {time_idx}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"histogram_time_{time_idx}.png"))
        plt.close()

    real_samples_plot = real_eval[:num_plot_samples]
    gen_samples_plot = gen_eval[:num_plot_samples]

    plt.figure()
    real_first = True
    gen_first = True
    for r in real_samples_plot:
        kwargs = {"label": "Real"} if real_first else {}
        plt.plot(ts.cpu(), r.cpu(), color="dodgerblue", linewidth=0.7, alpha=0.8, **kwargs)
        real_first = False

    for g in gen_samples_plot:
        kwargs = {"label": "Generated"} if gen_first else {}
        plt.plot(ts.cpu(), g.cpu(), color="crimson", linewidth=0.7, alpha=0.8, **kwargs)
        gen_first = False

    plt.title(f"Comparison of {num_plot_samples} Real/Generated samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "samples_comparison.png"))
    plt.close()

"""Core SDE-GAN training loop with SWA and post-hoc diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.swa_utils as swa_utils
import torchcde
import tqdm

from sdgan.discriminator import Discriminator
from sdgan.evaluation import evaluate_loss, evaluate_metrics
from sdgan.generator import Generator
from sdgan.plotting import plot
from sdgan.preprocess import normalize_features_by_initial_value


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters controlling optimization and discriminator choice."""

    initial_noise_size: int = 5
    noise_size: int = 3
    hidden_size: int = 16
    mlp_size: int = 16
    num_layers: int = 1
    generator_lr: float = 2e-4
    discriminator_lr: float = 1e-3
    steps: int = 10000
    init_mult1: float = 3.0
    init_mult2: float = 0.5
    weight_decay: float = 0.01
    swa_step_start: int = 5000
    steps_per_print: int = 5000
    batch_size: int = 1024
    discriminator_type: str = "type5"
    num_terms: int = 3
    num_plot_samples: int = 1000
    plot_locs: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)


def train_sde_gan(
    ts: torch.Tensor,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    data_channels: int,
    *,
    train_config: TrainConfig,
    device: torch.device | str,
    save_dir: str,
) -> tuple[Generator, Discriminator]:
    """Runs alternating generator/discriminator optimization with diagnostics.

    Args:
        ts: Time grid tensor ``(T,)`` residing on ``device``.
        train_dataloader: Batches shaped like ``torchcde.linear_interpolation_coeffs`` outputs.
        eval_dataloader: Held-out coefficients for plotting and metrics (can reuse train splits).
        data_channels: Feature dimensions excluding the leading time channel.
        train_config: Optimization and architecture knobs.
        device: Target device string or ``torch.device``.
        save_dir: Directory receiving plots, metrics, and sample ``.npy`` exports.

    Returns:
        Tuple of trained ``Generator`` and ``Discriminator`` with SWA weights loaded.
    """
    device_obj = torch.device(device)
    ts = ts.to(device_obj)

    infinite_train_dataloader = (
        elem for it in iter(lambda: train_dataloader, None) for elem in it
    )

    generator = Generator(
        data_size=data_channels,
        initial_noise_size=train_config.initial_noise_size,
        noise_size=train_config.noise_size,
        hidden_size=train_config.hidden_size,
        mlp_size=train_config.mlp_size,
        num_layers=train_config.num_layers,
    ).to(device_obj)

    discriminator = Discriminator(
        data_size=data_channels,
        hidden_size=train_config.hidden_size,
        mlp_size=train_config.mlp_size,
        num_layers=train_config.num_layers,
        discriminator_type=train_config.discriminator_type,
        num_terms=train_config.num_terms,
    ).to(device_obj)

    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    with torch.no_grad():
        for param in generator._initial.parameters():
            param.mul_(train_config.init_mult1)
        for param in generator._func.parameters():
            param.mul_(train_config.init_mult2)

    generator_optim = torch.optim.Adadelta(
        generator.parameters(),
        lr=train_config.generator_lr,
        weight_decay=train_config.weight_decay,
    )
    discriminator_optim = torch.optim.Adadelta(
        discriminator.parameters(),
        lr=train_config.discriminator_lr,
        weight_decay=train_config.weight_decay,
    )

    trange = tqdm.tqdm(range(train_config.steps))
    loss_history: list[float] = []
    real_score_history: list[float] = []
    gen_score_history: list[float] = []

    for step in trange:
        real_samples, = next(infinite_train_dataloader)
        real_samples = real_samples.to(device_obj)

        gen_samples = generator(ts, train_config.batch_size)
        if train_config.discriminator_type == "type5":
            gen_samples = normalize_features_by_initial_value(gen_samples)
            gen_score = discriminator(gen_samples)
        else:
            gen_score = discriminator(gen_samples)
        real_score = discriminator(real_samples)

        loss = gen_score - real_score
        loss.backward()

        for param in generator.parameters():
            if param.grad is not None:
                param.grad.mul_(-1)

        generator_optim.step()
        discriminator_optim.step()
        generator_optim.zero_grad()
        discriminator_optim.zero_grad()

        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

        if step > train_config.swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

        if (step % train_config.steps_per_print) == 0 or step == train_config.steps - 1:
            total_loss, r_score, g_score = evaluate_loss(
                ts,
                train_config.batch_size,
                train_dataloader,
                generator,
                discriminator,
                step,
            )
            loss_history.append(total_loss)
            real_score_history.append(r_score)
            gen_score_history.append(g_score)

            if step > train_config.swa_step_start:
                avg_loss, _, _ = evaluate_loss(
                    ts,
                    train_config.batch_size,
                    train_dataloader,
                    averaged_generator.module,
                    averaged_discriminator.module,
                    step,
                )
                trange.write(
                    f"Step {step} | Loss unavg: {total_loss:.4f}, Loss avg: {avg_loss:.4f}"
                )
            else:
                trange.write(f"Step {step} | Loss unavg: {total_loss:.4f}")

    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())

    os.makedirs(save_dir, exist_ok=True)

    plot(
        ts,
        generator,
        eval_dataloader,
        train_config.num_plot_samples,
        list(train_config.plot_locs),
        save_dir,
    )

    plt.figure()
    plt.plot(loss_history, label="Unaveraged Loss")
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Unaveraged Loss Over Training Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_over_steps.png"))
    plt.close()

    plt.figure()
    plt.plot(real_score_history, label="Real Score", color="green")
    plt.plot(gen_score_history, label="Generated Score", color="red")
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Discriminator Scores Over Training Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "disc_scores_over_steps.png"))
    plt.close()

    ds_len = len(eval_dataloader.dataset)
    metrics_result = evaluate_metrics(
        ts,
        generator,
        eval_dataloader,
        device=device_obj,
        num_samples=min(1024, ds_len),
        num_bins=50,
        tail_percentiles=[0.01, 0.05],
    )
    print("==== Evaluation Metrics ====")
    print(f"  MISE: {metrics_result['MISE']:.6f}")
    for k, v in metrics_result["TailDiff"].items():
        print(f"  {k}: {v:.6f}")

    metrics_path = os.path.join(save_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"MISE: {metrics_result['MISE']:.6f}\n")
        for k, v in metrics_result["TailDiff"].items():
            f.write(f"{k}: {v:.6f}\n")

    real_batch, = next(iter(eval_dataloader))
    real_batch = real_batch[:10].to(device_obj)
    real_eval = torchcde.LinearInterpolation(real_batch).evaluate(ts)
    real_eval_np = real_eval[..., 1].cpu().numpy()

    with torch.no_grad():
        gen_batch = generator(ts, real_batch.size(0)).detach().cpu()
    gen_eval = torchcde.LinearInterpolation(gen_batch).evaluate(ts.detach().cpu())
    gen_eval_np = gen_eval[..., 1].numpy()

    np.save(os.path.join(save_dir, "real_samples.npy"), real_eval_np)
    np.save(os.path.join(save_dir, "generated_samples.npy"), gen_eval_np)

    print(f"Real samples & generated samples saved to: {save_dir}")
    return generator, discriminator

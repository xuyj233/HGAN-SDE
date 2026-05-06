"""Loads coefficient tensors and starts :func:`sdgan.training.train_sde_gan`."""

from __future__ import annotations

import argparse

import torch

from sdgan.training import TrainConfig, train_sde_gan


def _load_coeff_pack(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads ``ts`` and spline ``coeffs`` from a torch checkpoint.

    Args:
        path: Pickle path written via ``torch.save``.
        device: Target device for tensors.

    Returns:
        ``(ts, coeffs)`` where ``coeffs`` is ``(N, T, 1 + data_channels)``. If the file
        omits ``ts``, the first trajectory's time channel is used.

    Raises:
        ValueError: If required keys are missing or tensor ranks are invalid.
    """
    obj = torch.load(path, map_location=device, weights_only=False)
    if "coeffs" not in obj:
        raise ValueError(f"{path} must contain key 'coeffs' (tensor N x T x C).")
    coeffs = obj["coeffs"]
    if coeffs.dim() != 3:
        raise ValueError("`coeffs` must be a 3-D tensor (batch, time, channels).")
    coeffs = coeffs.to(device=device, dtype=torch.float32)
    ts = obj.get("ts")
    if ts is None:
        ts = coeffs[0, :, 0].contiguous().to(dtype=torch.float32)
    else:
        ts = ts.reshape(-1).to(device=device, dtype=torch.float32)
    if ts.numel() != coeffs.size(1):
        raise ValueError("Length of `ts` must match coeffs time dimension.")
    return ts, coeffs


def _build_loader(
    coeffs: torch.Tensor, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    ds = torch.utils.data.TensorDataset(coeffs)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main() -> None:
    """CLI: load training/eval coefficients and run the core training routine."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the SDE generator / Hermite-style discriminator on precomputed "
            "torchcde spline coefficients (no built-in SDE simulation)."
        )
    )
    parser.add_argument(
        "--train_pt",
        type=str,
        required=True,
        help="torch.save dict with 'coeffs' and optional 'ts' (see README).",
    )
    parser.add_argument(
        "--eval_pt",
        type=str,
        default="",
        help="Optional second pack for evaluation; defaults to --train_pt.",
    )
    parser.add_argument("--save_dir", type=str, default="demo")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--initial_noise_size", type=int, default=5)
    parser.add_argument("--noise_size", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--mlp_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--generator_lr", type=float, default=2e-4)
    parser.add_argument("--discriminator_lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--init_mult1", type=float, default=3.0)
    parser.add_argument("--init_mult2", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--swa_step_start", type=int, default=5000)
    parser.add_argument("--steps_per_print", type=int, default=5000)
    parser.add_argument("--num_plot_samples", type=int, default=1000)
    parser.add_argument(
        "--plot_locs",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    parser.add_argument("--discriminator_type", type=str, default="type5")
    parser.add_argument("--num_terms", type=int, default=3)

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    ts_train, train_coeffs = _load_coeff_pack(args.train_pt, device_obj)
    eval_path = args.eval_pt or args.train_pt
    _, eval_coeffs = _load_coeff_pack(eval_path, device_obj)

    data_channels = train_coeffs.size(-1) - 1
    if eval_coeffs.size(-1) - 1 != data_channels:
        raise ValueError("Train and eval coeffs must share feature channel count.")

    train_loader = _build_loader(train_coeffs, args.batch_size, shuffle=True)
    eval_loader = _build_loader(eval_coeffs, args.batch_size, shuffle=False)

    cfg = TrainConfig(
        initial_noise_size=args.initial_noise_size,
        noise_size=args.noise_size,
        hidden_size=args.hidden_size,
        mlp_size=args.mlp_size,
        num_layers=args.num_layers,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        steps=args.steps,
        init_mult1=args.init_mult1,
        init_mult2=args.init_mult2,
        weight_decay=args.weight_decay,
        swa_step_start=args.swa_step_start,
        steps_per_print=args.steps_per_print,
        batch_size=args.batch_size,
        discriminator_type=args.discriminator_type,
        num_terms=args.num_terms,
        num_plot_samples=args.num_plot_samples,
        plot_locs=tuple(args.plot_locs),
    )

    train_sde_gan(
        ts_train,
        train_loader,
        eval_loader,
        data_channels,
        train_config=cfg,
        device=device_obj,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

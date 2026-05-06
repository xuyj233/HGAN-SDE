# HGAN-SDE

Hermite / neural-CDE discriminators with an SDE generator (adjoint + `torchcde`).

This repository keeps **models and the training loop only**. SDE simulation, tabular
loading, and building `torchcde` coefficients from raw series live in your own code.

## Install

```bash
pip install -r requirements.txt
```

## Data format for the CLI

`python -m sdgan` expects a Torch file:

```python
import torch
import torchcde

# ys: float tensor (N, T, 1 + F) — column 0 is time, remainder are features (e.g. F = 1)
coeffs = torchcde.linear_interpolation_coeffs(ys)
torch.save({"ts": ys[0, :, 0], "coeffs": coeffs}, "train.pt")
torch.save({"ts": ys[0, :, 0], "coeffs": coeffs}, "eval.pt")  # optional hold-out
```

If you omit `"ts"`, the loader uses `coeffs[0, :, 0]` as the shared time grid.

## Train

From this directory (the folder that contains `sdgan/`):

```bash
PYTHONPATH=. python -m sdgan --train_pt train.pt --eval_pt eval.pt --save_dir demo
```

## Library API

Import `TrainConfig` and `train_sde_gan` from `sdgan` and pass your own
`torch.utils.data.DataLoader` returning batches compatible with
`torchcde.LinearInterpolation`.

Remote: [HGAN-SDE on GitHub](https://github.com/xuyj233/HGAN-SDE).

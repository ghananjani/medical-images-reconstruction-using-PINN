"""Reproducible supervised CT U-Net training, validation and best checkpointing."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .ct_dataset import SyntheticCTPairs
from .unet import CTUNet


@dataclass
class TrainConfig:
    output: str = "outputs/ct_unet"
    size: int = 32
    projections: int = 30
    angle_span: float = 180.0
    noise: float = 0.01
    train_cases: int = 24
    val_cases: int = 8
    test_cases: int = 8
    epochs: int = 20
    patience: int = 5
    batch_size: int = 4
    channels: int = 8
    lr: float = 0.001
    seed: int = 42
    device: str = "auto"
    resume: str = ""

    def validate(self):
        if self.size < 16 or self.size % 4:
            raise ValueError("image size must be >=16 and divisible by four")
        if min(self.train_cases, self.val_cases, self.test_cases, self.epochs,
               self.patience, self.batch_size) < 1:
            raise ValueError("dataset sizes, epochs, patience and batch size must be positive")
        if self.channels < 4 or self.channels % 4 or self.lr <= 0:
            raise ValueError("invalid U-Net width or learning rate")
        if not 2 <= self.projections <= 720 or not 0 < self.angle_span <= 180:
            raise ValueError("invalid CT geometry")
        if self.noise < 0 or self.seed < 0 or self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("invalid noise, seed or device")
        if self.seed >= 100_000:
            raise ValueError("seed must be <100000 for disjoint dataset seed ranges")


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _make_loader(cfg, n, offset, shuffle, epoch=0):
    dataset = SyntheticCTPairs(n=n, size=cfg.size, projections=cfg.projections,
                               angle_span=cfg.angle_span, noise=cfg.noise,
                               seed_offset=offset)
    generator = torch.Generator().manual_seed(cfg.seed + 1009 * epoch + 7)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def _epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for input_image, target in loader:
        input_image = input_image.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(input_image)
                loss = loss_fn(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
        total += float(loss.detach().item()) * input_image.shape[0]
        count += input_image.shape[0]
    return total / count


def train_ct_unet(cfg: TrainConfig):
    """Train and save best validation checkpoint, never selected on test cases."""
    cfg.validate()
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    _seed_all(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                          else "cpu" if cfg.device == "auto" else cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_seed = 100_000 + cfg.seed
    val_seed = 200_000 + cfg.seed
    test_seed = 300_000 + cfg.seed
    val_loader = _make_loader(cfg, cfg.val_cases, val_seed, False)
    # The test set is deliberately not accessed during optimization.
    model = CTUNet(cfg.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, cfg.patience // 2))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.L1Loss()
    best = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    checkpoint = out / "best.pt"
    last_checkpoint = out / "last.pt"
    start_epoch = 1
    if cfg.resume:
        resume_path = Path(cfg.resume).resolve()
        if resume_path != last_checkpoint.resolve():
            raise ValueError("resume must be the last.pt in the configured output folder")
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        previous = dict(saved["config"])
        current = asdict(cfg)
        # Expanding the epoch budget is allowed; changing data/model/optimizer isn't.
        for key in current.keys() - {"epochs", "output", "device", "resume"}:
            if current[key] != previous[key]:
                raise ValueError(f"incompatible resume configuration: {key}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        if "scaler_state" in saved:
            scaler.load_state_dict(saved["scaler_state"])
        best = float(saved["best_val_mae"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        if start_epoch > cfg.epochs:
            raise ValueError("resume target epochs must exceed checkpoint epoch")
        if bad_epochs >= cfg.patience:
            raise ValueError("cannot resume a run that already early-stopped")
        if not checkpoint.is_file():
            raise ValueError("best.pt is missing alongside last.pt")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loader = _make_loader(cfg, cfg.train_cases, train_seed, True, epoch=epoch)
        train_loss = _epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        val_loss = _epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_mae": train_loss,
                        "val_mae": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        if val_loss < best - 1e-7:
            best = val_loss
            best_epoch = epoch
            bad_epochs = 0
            temp = out / "best.pt.tmp"
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                        "epoch": epoch, "best_val_mae": best,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "seed_splits": {"train": train_seed, "val": val_seed,
                                        "test": test_seed}}, temp)
            os.replace(temp, checkpoint)
        else:
            bad_epochs += 1
        # Save latest optimizer/scheduler state even when validation worsens.
        # Keep the best checkpoint independently for held-out evaluation.
        latest = {
            "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "config": asdict(cfg), "epoch": epoch, "best_epoch": best_epoch,
            "best_val_mae": best, "bad_epochs": bad_epochs, "history": history,
            "seed_splits": {"train": train_seed, "val": val_seed, "test": test_seed},
        }
        temp = out / "last.pt.tmp"
        torch.save(latest, temp)
        os.replace(temp, last_checkpoint)
        if bad_epochs >= cfg.patience:
            break

    with (out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["epoch"] for r in history], [r["train_mae"] for r in history],
            label="Train MAE")
    ax.plot([r["epoch"] for r in history], [r["val_mae"] for r in history],
            label="Validation MAE")
    ax.set(xlabel="Epoch", ylabel="L1 loss", title="Synthetic CT U-Net training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=160)
    plt.close(fig)
    manifest = {"config": asdict(cfg), "best_epoch": best_epoch,
                "best_val_mae": best, "epochs_completed": len(history),
                "early_stopped": bad_epochs >= cfg.patience, "device": str(device),
                "resumed_from": cfg.resume or None, "checkpoint_last": str(last_checkpoint),
                "torch": torch.__version__, "numpy": np.__version__,
                "python": platform.python_version(), "git_commit": _git_sha(),
                "split_seed_offsets": {"train": train_seed, "val": val_seed,
                                       "test": test_seed},
                "note": "Synthetic affine-perturbed Shepp-Logan, research-only"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_ct_unet(checkpoint: str | Path, device: str = "cpu"):
    """Load only trusted locally-generated checkpoints; never untrusted pickle files."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = CTUNet(int(state["config"]["channels"])).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state["config"]


def main():
    parser = argparse.ArgumentParser(description="Train synthetic CT FBP-to-image U-Net")
    for key, default in asdict(TrainConfig()).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default),
                            default=default)
    args = parser.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    manifest = train_ct_unet(cfg)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

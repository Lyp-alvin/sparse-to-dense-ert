from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from Dataset import Stage1WADataset, Stage1WARhoaToWADataset
from UNetPP import UNetPP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


@torch.no_grad()
def generate_stage1_wa_predictions(
    data_root: str | Path,
    split: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    device: torch.device,
    pred_prefix: str,
) -> None:
    output_dir = ensure_dir(output_dir)
    dataset = Stage1WADataset(data_root, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNetPP(in_channels=1, out_channels=1).to(device)
    load_checkpoint(Path(checkpoint_path), model, map_location=device)
    model.eval()

    for batch in dataloader:
        sample_ids = batch["sample_id"]
        inputs = batch["WA_sparse"].to(device)
        preds = model(inputs).cpu().numpy()
        for idx_in_batch, sample_id in enumerate(sample_ids):
            sample_id_int = int(sample_id)
            save_path = output_dir / f"{pred_prefix}_{sample_id_int}.mat"
            if save_path.exists():
                continue
            sio.savemat(
                save_path,
                {
                    "pred_WA_dense": preds[idx_in_batch, 0, :, :].astype(np.float32),
                },
            )


@torch.no_grad()
def generate_stage1_wa_rhoa_predictions(
    data_root: str | Path,
    split: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    device: torch.device,
    pred_prefix: str,
) -> None:
    output_dir = ensure_dir(output_dir)
    dataset = Stage1WARhoaToWADataset(data_root, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNetPP(in_channels=2, out_channels=1).to(device)
    load_checkpoint(Path(checkpoint_path), model, map_location=device)
    model.eval()

    for batch in dataloader:
        sample_ids = batch["sample_id"]
        inputs = torch.cat([batch["WA_sparse"], batch["rhoa_sparse"]], dim=1).to(device)
        preds = model(inputs).cpu().numpy()
        for idx_in_batch, sample_id in enumerate(sample_ids):
            sample_id_int = int(sample_id)
            save_path = output_dir / f"{pred_prefix}_{sample_id_int}.mat"
            if save_path.exists():
                continue
            sio.savemat(
                save_path,
                {
                    "pred_WA_dense": preds[idx_in_batch, 0, :, :].astype(np.float32),
                },
            )

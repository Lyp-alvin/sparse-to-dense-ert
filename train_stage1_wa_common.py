from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from Dataset import Stage1WADataset, Stage1WARhoaToWADataset
from UNetPP import UNetPP
from pipeline_utils import create_logger, ensure_dir, load_checkpoint, save_checkpoint, set_seed


def build_parser(description: str, default_work_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", default="data_chaofen")
    parser.add_argument("--work-dir", default=default_work_dir)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", default="")
    return parser


def run_epoch(
    model: UNetPP,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    mode: str,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        if mode == "fangan1":
            inputs = batch["WA_sparse"].to(device)
        elif mode == "fangan5":
            inputs = torch.cat([batch["WA_sparse"], batch["rhoa_sparse"]], dim=1).to(device)
        else:
            raise ValueError(f"Unsupported Stage-1 mode: {mode}")
        wa_dense = batch["WA_dense"].to(device)
        preds = model(inputs)
        loss = F.mse_loss(preds, wa_dense)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / total_batches


def run_training(args: argparse.Namespace, suffix: str, mode: str = "fangan1") -> None:
    set_seed(args.seed)
    work_dir = ensure_dir(args.work_dir)
    logger = create_logger(f"stage1_train_{suffix}", work_dir / f"train_stage1_{suffix}.log")
    loss_file = work_dir / f"loss_stage1_{suffix}.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if mode == "fangan1":
        dataset_class = Stage1WADataset
        in_channels = 1
    elif mode == "fangan5":
        dataset_class = Stage1WARhoaToWADataset
        in_channels = 2
    else:
        raise ValueError(f"Unsupported Stage-1 mode: {mode}")

    train_dataset = dataset_class(args.data_root, "train")
    val_dataset = dataset_class(args.data_root, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNetPP(in_channels=in_channels, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.patience,
        factor=args.factor,
        threshold=args.threshold,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    if not loss_file.exists() or start_epoch == 1:
        loss_file.write_text("epoch,train_loss,val_loss,lr\n", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, mode)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, None, mode)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %03d | train_loss=%.6f | val_loss=%.6f | lr=%.6e",
            epoch,
            train_loss,
            val_loss,
            current_lr,
        )
        with loss_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{current_lr:.8e}\n")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(work_dir / f"best_stage1_{suffix}.pth", epoch, model, optimizer, scheduler, best_val_loss)
            logger.info("Updated best checkpoint at epoch %03d with val_loss=%.6f", epoch, val_loss)
        save_checkpoint(work_dir / f"last_stage1_{suffix}.pth", epoch, model, optimizer, scheduler, best_val_loss)

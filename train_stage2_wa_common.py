from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from Dataset import Stage2Fangan1Dataset, Stage2Fangan2Dataset, discover_split_ids
from UNetPP import UNetPP
from pipeline_utils import (
    create_logger,
    ensure_dir,
    generate_stage1_wa_rhoa_predictions,
    generate_stage1_wa_predictions,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def build_parser(description: str, default_work_dir: str, default_stage1_work_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", default="data_chaofen")
    parser.add_argument("--work-dir", default=default_work_dir)
    parser.add_argument("--stage1-work-dir", default=default_stage1_work_dir)
    parser.add_argument("--stage1-checkpoint", default="")
    parser.add_argument("--stage1-checkpoint-name", default="")
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
    parser.add_argument("--reuse-stage1-predictions", action="store_true",
                        help="Reuse this run's existing Stage-1 cache when resuming.")
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
        if mode in {"fangan1", "fangan5"}:
            inputs = torch.cat([batch["pred_WA_dense"], batch["WA_sparse"]], dim=1).to(device)
        elif mode == "fangan2":
            inputs = batch["pred_WA_dense"].to(device)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        rho = batch["rho"].to(device)
        preds = model(inputs)
        loss = F.mse_loss(preds, rho)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / total_batches


def run_training(args: argparse.Namespace, suffix: str, mode: str) -> None:
    set_seed(args.seed)
    work_dir = ensure_dir(args.work_dir)
    stage1_work_dir = Path(args.stage1_work_dir)
    if args.stage1_checkpoint:
        stage1_checkpoint = Path(args.stage1_checkpoint)
    else:
        checkpoint_name = args.stage1_checkpoint_name or f"best_stage1_{suffix}.pth"
        stage1_checkpoint = stage1_work_dir / checkpoint_name
    stage1_pred_train_dir = work_dir / "stage1_predictions" / "train"
    stage1_pred_val_dir = work_dir / "stage1_predictions" / "val"
    pred_prefix = f"pred_stage1_{suffix}"

    logger = create_logger(f"stage2_train_{suffix}", work_dir / f"train_stage2_{suffix}.log")
    loss_file = work_dir / f"loss_stage2_{suffix}.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if args.reuse_stage1_predictions:
        if not args.resume or not stage1_checkpoint.is_file():
            raise ValueError("Reusing Stage-1 predictions requires a resumed run and its Stage-1 checkpoint.")
        logger.info("Reusing Stage-1 predictions in %s", work_dir / "stage1_predictions")
    elif mode == "fangan5":
        generate_stage1_wa_rhoa_predictions(
            args.data_root,
            "train",
            stage1_checkpoint,
            stage1_pred_train_dir,
            args.batch_size,
            device,
            pred_prefix,
        )
        generate_stage1_wa_rhoa_predictions(
            args.data_root,
            "val",
            stage1_checkpoint,
            stage1_pred_val_dir,
            args.batch_size,
            device,
            pred_prefix,
        )
    else:
        generate_stage1_wa_predictions(args.data_root, "train", stage1_checkpoint, stage1_pred_train_dir, args.batch_size, device, pred_prefix)
        generate_stage1_wa_predictions(args.data_root, "val", stage1_checkpoint, stage1_pred_val_dir, args.batch_size, device, pred_prefix)

    if mode in {"fangan1", "fangan5"}:
        train_dataset = Stage2Fangan1Dataset(args.data_root, "train", stage1_pred_train_dir, pred_prefix=pred_prefix)
        val_dataset = Stage2Fangan1Dataset(args.data_root, "val", stage1_pred_val_dir, pred_prefix=pred_prefix)
        in_channels = 2
    elif mode == "fangan2":
        train_dataset = Stage2Fangan2Dataset(args.data_root, "train", stage1_pred_train_dir, pred_prefix=pred_prefix)
        val_dataset = Stage2Fangan2Dataset(args.data_root, "val", stage1_pred_val_dir, pred_prefix=pred_prefix)
        in_channels = 1
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if args.reuse_stage1_predictions:
        for split, dataset in (("train", train_dataset), ("val", val_dataset)):
            if dataset.ids != discover_split_ids(args.data_root, split):
                raise ValueError(f"Incomplete Stage-1 prediction cache for {split}")

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
        checkpoint = load_checkpoint(Path(args.resume), model, optimizer, scheduler, map_location=device)
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
            save_checkpoint(work_dir / f"best_stage2_{suffix}.pth", epoch, model, optimizer, scheduler, best_val_loss)
            logger.info("Updated best checkpoint at epoch %03d with val_loss=%.6f", epoch, val_loss)
        save_checkpoint(work_dir / f"last_stage2_{suffix}.pth", epoch, model, optimizer, scheduler, best_val_loss)

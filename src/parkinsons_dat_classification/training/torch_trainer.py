"""PyTorch training and validation loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from parkinsons_dat_classification.config import TorchTrainingConfig
from parkinsons_dat_classification.training.metrics import compute_binary_classification_metrics, BinaryClassificationMetrics
from parkinsons_dat_classification.training.torch_checkpointing import save_torch_checkpoint

LOGGER = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool = True,
) -> float:
    """Train the model for a single epoch."""
    
    model.train()
    total_loss = 0.0
    
    # Disable AMP on CPU
    if device.type == "cpu":
        amp = False
        
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool = True,
) -> tuple[float, BinaryClassificationMetrics]:
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_scores = []
    all_labels = []
    
    if device.type == "cpu":
        amp = False
    
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)
        
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
            
        total_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        
        # Ensure we always get an iterable list regardless of batch size
        all_preds.extend(preds.view(-1).cpu().tolist())
        all_scores.extend(probs.view(-1).cpu().tolist())
        all_labels.extend(labels.view(-1).cpu().tolist())
        
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_binary_classification_metrics(all_labels, all_preds, all_scores)
    return avg_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TorchTrainingConfig,
    checkpoint_dir: Path,
    summary_writer: SummaryWriter | None = None,
) -> dict[str, Any]:
    """Full PyTorch training pipeline."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    best_roc_auc = -1.0
    best_metrics = None
    
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config.amp)
        val_loss, metrics = validate(model, val_loader, loss_fn, device, config.amp)
        
        LOGGER.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ROC AUC: {metrics.roc_auc:.4f} | "
            f"Val F1: {metrics.f1:.4f}"
        )
        scheduler.step()
        
        if summary_writer is not None:
            summary_writer.add_scalar("Loss/train", train_loss, epoch)
            summary_writer.add_scalar("Loss/val", val_loss, epoch)
            summary_writer.add_scalar("Metrics/val_roc_auc", metrics.roc_auc, epoch)
            summary_writer.add_scalar("Metrics/val_f1", metrics.f1, epoch)
            summary_writer.add_scalar("Metrics/val_accuracy", metrics.accuracy, epoch)

        if metrics.roc_auc > best_roc_auc:
            best_roc_auc = metrics.roc_auc
            best_metrics = metrics
            
            metadata = {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "roc_auc": metrics.roc_auc,
                "f1": metrics.f1,
            }
            save_torch_checkpoint(model, checkpoint_dir, metadata, "best_model.pt")
            
    return {"best_metrics": best_metrics}

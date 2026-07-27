"""Checkpointing logic for PyTorch models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_torch_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    filename: str = "model.pt",
) -> Path:
    """Save a PyTorch model and its metadata sidecar."""
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoint_dir / filename
    meta_path = checkpoint_dir / f"{model_path.stem}_meta.json"
    
    torch.save(model.state_dict(), model_path)
    
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    return model_path


def load_torch_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a PyTorch model's state_dict and its metadata."""
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    meta_path = checkpoint_path.parent / f"{checkpoint_path.stem}_meta.json"
    metadata = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    return model, metadata

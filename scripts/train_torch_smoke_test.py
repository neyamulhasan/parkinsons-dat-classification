"""Smoke test for PyTorch training pipeline."""

import logging
from pathlib import Path
import shutil

from parkinsons_dat_classification.config import TorchTrainingConfig
from parkinsons_dat_classification.data.dataset import DaTSample, build_torch_dataloaders
from parkinsons_dat_classification.models.simple_3d import build_3d_cnn
from parkinsons_dat_classification.training.torch_trainer import train_model
from parkinsons_dat_classification.transforms.transforms import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO)

def run_smoke_test():
    scan_root = Path("data/smoke_test/smoke_test_data_2gePzfM/niftis")
    if not scan_root.exists():
        logging.warning("Smoke test data not found.")
        return

    uids = [p.name[:-7] for p in scan_root.glob("*.nii.gz")]
    if not uids:
        logging.warning("No NIfTI files found.")
        return

    # Use first 10 for train, next 10 for val
    train_samples = [DaTSample(uid=uid, scan_path=scan_root / f"{uid}.nii.gz", label=i % 2) for i, uid in enumerate(uids[:10])]
    val_samples = [DaTSample(uid=uid, scan_path=scan_root / f"{uid}.nii.gz", label=i % 2) for i, uid in enumerate(uids[10:20])]

    train_loader, val_loader = build_torch_dataloaders(
        train_samples,
        val_samples,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms(),
        batch_size=2,
        num_workers=0
    )

    model = build_3d_cnn(model_name="densenet121")
    config = TorchTrainingConfig(
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cpu", # Force CPU for smoke test to avoid CUDA issues
        amp=False
    )
    checkpoint_dir = Path("outputs/smoke_test_checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    logging.info("Starting PyTorch training smoke test...")
    results = train_model(model, train_loader, val_loader, config, checkpoint_dir)
    logging.info(f"Smoke test completed. Best Val ROC AUC: {results['best_metrics'].roc_auc}")

if __name__ == "__main__":
    run_smoke_test()

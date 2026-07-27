"""Main entry point for PyTorch deep learning training."""

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from parkinsons_dat_classification.config import build_app_config
from parkinsons_dat_classification.data.dataset import build_samples, split_samples, build_torch_dataloaders
from parkinsons_dat_classification.models.simple_3d import build_3d_cnn
from parkinsons_dat_classification.training.torch_trainer import train_model
from parkinsons_dat_classification.transforms.transforms import get_train_transforms, get_val_transforms
from parkinsons_dat_classification.utils.logging import configure_logging
from parkinsons_dat_classification.submission.pipeline import build_torch_submission_predictions, write_submission_csv

LOGGER = logging.getLogger(__name__)

def main() -> int:
    """Run the PyTorch training pipeline on the full dataset."""
    configure_logging()
    
import torch
    # Resolve device string; support 'auto' to select CUDA if available, else CPU
    device_str = config.torch_training.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    config.torch_training.device = device_str
    
    # We load real training samples
    labels_csv = config.paths.raw_data / "train_labels.csv"
    scan_dir = config.paths.raw_data / "train_images" / "niftis"
    
    # If the real dataset is missing, we fallback to smoke test data for the script to run cleanly
    if not labels_csv.exists() or not scan_dir.exists():
        LOGGER.warning("Real dataset not found in %s. Falling back to smoke test data...", config.paths.raw_data)
        
        # Smoke test fallback
        scan_dir = config.paths.smoke_test / "smoke_test_data_2gePzfM" / "niftis"
        if not scan_dir.exists():
            LOGGER.error("Smoke test data not found. Exiting.")
            return 1
            
        # Manually create labels since there is no train_labels.csv in smoke test
        from parkinsons_dat_classification.data.dataset import DaTSample
        uids = [p.name[:-7] for p in scan_dir.glob("*.nii.gz")]
        all_samples = [DaTSample(uid=uid, scan_path=scan_dir / f"{uid}.nii.gz", label=i % 2) for i, uid in enumerate(uids)]
    else:
        LOGGER.info("Loading real dataset from %s", labels_csv)
        all_samples = build_samples(labels_csv, scan_dir)
        
    train_samples, val_samples = split_samples(all_samples, validation_fraction=config.training.validation_fraction, seed=config.training.random_state)
    
    LOGGER.info("Training on %d samples, Validating on %d samples.", len(train_samples), len(val_samples))
    
    train_loader, val_loader = build_torch_dataloaders(
        train_samples,
        val_samples,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms(),
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers
    )
    
    model = build_3d_cnn("densenet121")
    
    tensorboard_dir = config.paths.logs / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    checkpoint_dir = config.artifacts.checkpoint_dir / "torch_models"
    
    LOGGER.info("Starting PyTorch training...")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.torch_training,
        checkpoint_dir=checkpoint_dir,
        summary_writer=writer,
    )
    
    writer.close()
    
    LOGGER.info("Training complete. Best Val ROC AUC: %.4f", results["best_metrics"].roc_auc)
    
    LOGGER.info("Generating submission for smoke test scans using best model...")
    predictions = build_torch_submission_predictions(
        model_name="densenet121",
        checkpoint_dir=checkpoint_dir,
        scan_root=config.paths.smoke_test / "smoke_test_data_2gePzfM" / "niftis",
        device=config.torch_training.device,
        batch_size=1
    )
    
    output_path = write_submission_csv(
        predictions,
        config.paths.outputs / "torch_submission.csv",
    )
    LOGGER.info("Submission written to %s", output_path)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

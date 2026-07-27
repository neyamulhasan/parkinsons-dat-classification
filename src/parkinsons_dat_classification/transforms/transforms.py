"""MONAI-based data augmentations and preprocessing."""

from __future__ import annotations

import monai.transforms as mt


def get_train_transforms() -> mt.Compose:
    """Get the training preprocessing and augmentation pipeline."""
    return mt.Compose([
        mt.LoadImaged(keys=["image"], image_only=True),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Resized(keys=["image"], spatial_size=(128, 128, 96)),
        mt.ScaleIntensityd(keys=["image"]),
        mt.RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
        mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    ])


def get_val_transforms() -> mt.Compose:
    """Get the validation and inference preprocessing pipeline."""
    return mt.Compose([
        mt.LoadImaged(keys=["image"], image_only=True),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Resized(keys=["image"], spatial_size=(128, 128, 96)),
        mt.ScaleIntensityd(keys=["image"]),
    ])

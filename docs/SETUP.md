# Environment Setup

This project uses PyTorch and medical imaging libraries (`nibabel`, `monai`). The exact setup depends on your available hardware (CPU vs. GPU).

## Prerequisites

- Python 3.12 or newer
- `pip` or a virtual environment manager of your choice

## Standard Installation

To install the project and its dependencies (including `torch`, `torchvision`, and `monai`) using the default Python package index:

```bash
make setup
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## GPU-Specific PyTorch Installation

If you are using an NVIDIA GPU, you may need to install the CUDA-enabled version of PyTorch explicitly before installing the project, depending on your system configuration.

For example, for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

See the [official PyTorch getting started guide](https://pytorch.org/get-started/locally/) for your specific hardware.

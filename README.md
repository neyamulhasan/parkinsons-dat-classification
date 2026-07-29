Parkinson's DaT Classification
==============================

A production-grade deep learning framework for automated classification of
Parkinsonian syndrome from Dopamine Transporter Single Photon Emission
Computed Tomography (DaT SPECT) brain imaging.

This repository is developed for the DrivenData DaT Parkinson's Challenge and
serves as a reproducible, modular, and research-oriented machine learning
project. The objective is to develop robust computer vision models for
classifying DaT SPECT scans while following modern software engineering
practices and the official competition requirements.


Project Objective
-----------------

The goal of this project is to design, implement, and evaluate an end-to-end
deep learning pipeline capable of classifying DaT SPECT brain scans into:

* Normal
* Abnormal (Parkinsonian Syndrome)

The project emphasizes:

* Reproducible machine learning
* Clean software architecture
* Modular design
* Experiment management
* Production-quality inference
* Competition-compliant submission


Repository Goals
----------------

* Build a configurable medical imaging pipeline.
* Support multiple deep learning architectures.
* Enable reproducible experimentation.
* Implement automated evaluation and inference.
* Produce a submission package compatible with the official competition runtime.
* Maintain high code quality through testing, documentation, and continuous improvement.


Current Foundation
------------------

The repository currently includes an initial Python package scaffold, YAML
configuration loading, and dataset inspection utilities.

Key entry points:

* configs/default.yaml
    Project paths and runtime defaults.

* scripts/inspect_dataset.py
    Local dataset-layout inspection.

* src/parkinsons_dat_classification/config.py
    Typed configuration loading.

* src/parkinsons_dat_classification/data/discovery.py
    Filesystem discovery.

* scripts/train.py
    Training entry point for PyTorch experiments.

* scripts/generate_submission.py
    Offline submission CSV generation (inference) using saved checkpoints.


Official Competition Runtime
----------------------------

The submission package targets the official DrivenData execution environment.

Programming Language
    Python 3.12

Execution
    Containerized

Entry Point
    `scripts/train.py` (training) and `scripts/generate_submission.py` (offline inference)

GPU
    NVIDIA A100 (80 GiB VRAM)

CPU
    24 AMD EPYC vCPUs

Memory
    220 GB RAM

CUDA
    12.9

Internet Access
    Not Available

Maximum Runtime
    3 Hours

Dependency Management
    Managed via `pyproject.toml` / setuptools. Install in a virtual env:

    pip install -e .

    Add optional packages as needed (e.g., OpenCV, Albumentations, pandas, matplotlib).

Submission Format
    ZIP archive containing inference code and model assets

The inference pipeline is designed to execute entirely offline within the
official runtime environment.


Project Technology Stack
------------------------

Programming

* Python 3.12

Deep Learning

* PyTorch
* TorchVision
* Automatic Mixed Precision (AMP)

Medical Imaging

* MONAI
* NiBabel

Computer Vision (optional)

* OpenCV (optional)
* Albumentations (optional)

Scientific Computing

* NumPy
* pandas
* SciPy
* scikit-learn

Visualization

* Matplotlib
* TensorBoard

Configuration

* YAML

The project uses `PyYAML` for config loading (`src/parkinsons_dat_classification/config.py`).

Experiment Management

* TensorBoard
* CSV Logging

Development Tools

* Git
* GitHub
* Docker
* pre-commit
* Ruff
* Black
* isort
* mypy
* pytest


Supported Model Architectures
-----------------------------

The framework is designed to support multiple deep learning models through a
modular model factory.

Current and planned architectures include:

* ResNet
* DenseNet
* EfficientNet
* ConvNeXt
* Vision Transformer (ViT)
* Swin Transformer

Additional architectures can be integrated with minimal code changes.


Repository Structure
--------------------

::

    parkinsons-dat-classification/

    ├── configs/                 Configuration files
    ├── data/                    Local dataset (excluded from Git)
    ├── scripts/                 Utility scripts
    ├── src/
    │   └── parkinsons_dat_classification/
    │       ├── config.py
    │       ├── data/
    │       └── utils/
    ├── tests/                   Automated tests
    ├── README.md
    ├── pyproject.toml
    └── .gitignore


Core Features
-------------

* Modular project architecture
* Configuration-driven training
* Automatic dataset discovery
* Data preprocessing and augmentation
* Cross-validation support
* Mixed precision training
* Checkpoint management
* Resume training
* Learning rate scheduling
* Comprehensive evaluation metrics
* Experiment tracking
* Offline inference pipeline
* Competition-compliant submission packaging
* Unit and integration testing
* Professional documentation


Dataset
-------

The competition dataset is not included in this repository.

To reproduce the experiments:

1. Register for the DrivenData DaT Parkinson's Challenge.
2. Accept the competition rules and data use agreement.
3. Download the dataset from the official competition website.
4. Store the dataset locally inside the data/ directory.

The dataset is subject to the competition's licensing and data use
restrictions and must not be redistributed.


Competition Compliance
----------------------

This repository is designed to comply with the official competition
requirements.

The repository includes:

* Source code
* Documentation
* Configuration files
* Training pipeline
* Evaluation pipeline
* Inference pipeline
* Submission packaging

The repository intentionally excludes:

* Competition datasets
* Protected metadata
* Hidden evaluation data
* Restricted competition resources


License
-------

License

No LICENSE file is present in the repository. If you intend to publish under the MIT License, add a `LICENSE` file containing the MIT license text. Until a license file is added, the repository does not have an explicit license declaration.

The competition dataset is not covered by the MIT License and remains subject
to the official competition terms and data use agreement.


Disclaimer
----------

This repository is an independent software implementation intended for
research, education, and participation in the DrivenData DaT Parkinson's
Challenge.

It is not a medical device and must not be used for clinical diagnosis or
patient care decisions.
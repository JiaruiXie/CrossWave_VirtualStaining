# CrossWave Virtual Staining

This repository contains the implementation of the virtual-staining case study used to evaluate **CrossWave**, a cross-wavelength knowledge-distillation framework for reduced-sensing microscopy.

The case study uses multispectral autofluorescence (MAF) breast-tissue images and an E-CycleGAN-based virtual H&E staining framework. The goal is to test whether information learned from richer multispectral measurements can be transferred to a reduced-sensing student that uses only the DAPI autofluorescence channel at inference.

## Overview

The experimental setup includes:

- a full-MAF E-CycleGAN teacher;
- a DAPI-only E-CycleGAN baseline;
- a CrossWave DAPI student;
- prediction-level knowledge distillation;
- feature-level knowledge distillation;
- distributional evaluation using FID and KID;

During CrossWave training, the teacher is pretrained and frozen. The student receives only the DAPI channel and is optimized using the original E-CycleGAN objective together with prediction- and feature-distillation losses.

## Repository Structure

```text
CrossWave_VirtualStaining/
├── configs/              # Training configuration files
├── data/
│   └── metadata/         # Dataset split metadata used by the experiments
├── scripts/              # Preprocessing, training, evaluation, and visualization scripts
├── src/
│   ├── data/             # Dataset and preprocessing utilities
│   ├── losses/           # E-CycleGAN and CrossWave loss functions
│   ├── models/           # Generator, discriminator, and E-CycleGAN models
│   ├── training/         # Training pipelines
│   └── utils/            # Configuration, seeding, and visualization utilities
├── .gitignore
├── README.md
└── requirements.txt
```

## Data

Raw microscopy images are **not included** in this repository. Please find the dataset at https://zenodo.org/records/17646116.

The repository retains the metadata required to reproduce the experimental split:

```text
data/metadata/data_split.csv
```

The original breast-tissue dataset contains multispectral autofluorescence images acquired with DAPI, SpGr, and SpOr channels together with conventional H&E images from the same tissue sections.

The E-CycleGAN training setup uses unpaired MAF and H&E image domains. For CrossWave distillation, teacher and student inputs are derived from spatially corresponding spectral measurements from the same MAF patch.

## Models

### Full-MAF Teacher

The teacher uses the richer multispectral autofluorescence representation and is trained using the E-CycleGAN virtual-staining framework.

### DAPI-only Baseline

The reduced-sensing baseline uses only the DAPI autofluorescence channel and is trained independently without teacher supervision.

### CrossWave Student

The CrossWave student uses DAPI only at inference and receives knowledge from the frozen MAF teacher during training through:

- **Prediction distillation:** L1 loss between student- and teacher-generated virtual H&E images.
- **Feature distillation:** L1 loss between intermediate student and teacher generator representations.

The student retains the original E-CycleGAN adversarial, cycle-consistency, feature-consistency, and saliency-consistency objectives.

## Installation

Create or activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The codebase uses PyTorch and was developed with CUDA-enabled training.

## Training

Training is controlled through YAML configuration files in `configs/`.

Example:

```powershell
& "C:\path\to\python.exe" `
    scripts\13_train_crosswave.py `
    --config "configs\crosswave_dapi_student.yaml"
```

Experiment-specific configuration files define:

- dataset paths and batch sizes;
- model capacity;
- E-CycleGAN penalty factors;
- CrossWave distillation weights;
- optimizer settings;
- adaptive learning-rate scheduling;
- FID/KID evaluation;
- checkpointing and early stopping.

## Virtual H&E Reconstruction

A trained CrossWave checkpoint can be used to reconstruct full virtual H&E ROIs from DAPI-only autofluorescence input.

Example:

```powershell
& "C:\path\to\python.exe" `
    scripts\15_reconstruct_subeval_crosswave_v3.py `
    --checkpoint "outputs\<experiment>\checkpoints\best_fid.pt" `
    --output-dir "outputs\<experiment>\sub_eval_vhe"
```

Generated TIFF images, checkpoints, caches, and other large outputs are excluded from version control.

## Evaluation

### Distributional Similarity

Virtual H&E outputs are compared with conventional H&E using:

- Fréchet Inception Distance (FID)
- Kernel Inception Distance (KID)

Lower values indicate greater similarity between generated and real H&E feature distributions.

## Citation

Citation information will be added upon publication of the CrossWave study.

## Acknowledgement

The breast-tissue virtual-staining dataset and E-CycleGAN baseline framework originate from the published multispectral autofluorescence breast-cancer virtual-staining study on which this case study builds.

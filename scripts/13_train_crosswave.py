"""
Train the CrossWave DAPI-only virtual-staining student.

Run from the project root:

    & "C:/Users/jiaru/.conda/envs/crosswave_vs/python.exe" `
      scripts\13_train_crosswave.py `
      --epochs 100

Resume:

    & "C:/Users/jiaru/.conda/envs/crosswave_vs/python.exe" `
      scripts\13_train_crosswave.py `
      --resume "outputs/crosswave_dapi_student/checkpoints/epoch_004.pt" `
      --epochs 96
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from src.training.trainer_crosswave import (
    CrossWaveTrainer,
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "crosswave_dapi_student.yaml"
)


def load_config(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(
            handle
        )


def set_seed(seed):
    seed = int(seed)

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Number of epochs to run from the current "
            "starting point."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a CrossWave STUDENT checkpoint to resume from. "
            "Do not pass the teacher checkpoint here."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            CONFIG_PATH
        ),
        help=(
            "CrossWave YAML configuration."
        ),
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            ROOT
            / config_path
        )

    config = load_config(
        config_path
    )

    experiment_cfg = config.get(
        "experiment",
        {},
    )

    if isinstance(
        experiment_cfg,
        dict,
    ):
        seed = experiment_cfg.get(
            "seed",
            config.get(
                "seed",
                42,
            ),
        )
    else:
        seed = config.get(
            "seed",
            42,
        )

    set_seed(
        seed
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.cuda.is_available(),
    )

    print(
        "Config:",
        config_path,
    )

    trainer = CrossWaveTrainer(
        config=config,
        root=ROOT,
    )

    if args.resume is not None:
        checkpoint_path = Path(
            args.resume
        )

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                ROOT
                / checkpoint_path
            )

        trainer.load_checkpoint(
            checkpoint_path
        )

    epochs = args.epochs

    if epochs is None:
        epochs = int(
            config.get(
                "training",
                {},
            ).get(
                "epochs",
                100,
            )
        )

    trainer.train(
        epochs=epochs
    )


if __name__ == "__main__":
    main()

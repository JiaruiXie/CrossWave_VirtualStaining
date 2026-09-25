from pathlib import Path
import argparse
import sys

import torch


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.utils.config import (
    load_config
)

from src.utils.seed import (
    set_seed
)

from src.training.trainer_ecyclegan import (
    ECycleGANTrainer
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "maf_teacher_stable_b_inverted_aux.yaml"
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
            "Path to checkpoint to resume from."
        ),
    )

    args = parser.parse_args()

    config = load_config(
        CONFIG_PATH
    )

    set_seed(
        config[
            "experiment"
        ][
            "seed"
        ]
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.cuda.is_available(),
    )

    trainer = ECycleGANTrainer(
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

    trainer.train(
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
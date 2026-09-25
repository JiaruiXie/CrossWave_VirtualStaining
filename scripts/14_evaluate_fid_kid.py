from pathlib import Path
import sys
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from torchmetrics.image.fid import (
    FrechetInceptionDistance,
)

from torchmetrics.image.kid import (
    KernelInceptionDistance,
)

from src.models.ecyclegan import ECycleGAN
from src.data.datasets import SingleDomainDataset
from src.utils.config import load_config


# =========================================================
# Paths
# =========================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "maf_teacher.yaml"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maf_teacher"
    / "checkpoints"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "maf_teacher"
    / "training_history.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maf_teacher"
    / "evaluation"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "fid_kid_results.csv"
)


# =========================================================
# Settings
# =========================================================

BATCH_SIZE = 16
NUM_WORKERS = 4

KID_SUBSETS = 50
KID_SUBSET_SIZE = 1000

SEED = 42


# =========================================================
# Reproducibility
# =========================================================

def set_seed(
    seed,
):
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


# =========================================================
# Convert images for TorchMetrics
# =========================================================

def to_metric_image(
    x,
):
    """
    Convert model tensor:

        [-1, 1]

    to TorchMetrics input:

        float32 [0, 1]

    Shape remains:

        B x 3 x H x W
    """

    x = (
        x + 1.0
    ) / 2.0

    x = torch.clamp(
        x,
        0.0,
        1.0,
    )

    return x.float()


# =========================================================
# Model loading
# =========================================================

def get_epoch(
    checkpoint_path,
    checkpoint,
):

    if isinstance(
        checkpoint,
        dict,
    ):

        for key in [
            "epoch",
            "completed_epoch",
        ]:

            if key in checkpoint:
                return int(
                    checkpoint[
                        key
                    ]
                )

    match = re.search(
        r"epoch_(\d+)",
        checkpoint_path.name,
    )

    if match:
        return int(
            match.group(1)
        )

    return None


def load_model(
    checkpoint_path,
    config,
    device,
):

    model_cfg = config[
        "model"
    ]

    model = ECycleGAN(

        x_channels=
            model_cfg[
                "x_channels"
            ],

        y_channels=
            model_cfg[
                "y_channels"
            ],

        base_channels=
            model_cfg[
                "base_channels"
            ],

        num_res_blocks=
            model_cfg[
                "num_res_blocks"
            ],

        padding_type=
            model_cfg[
                "padding_type"
            ],

    ).to(
        device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and "model"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    epoch = get_epoch(
        checkpoint_path,
        checkpoint,
    )

    return (
        model,
        epoch,
    )


# =========================================================
# History
# =========================================================

def load_history():

    if not HISTORY_PATH.exists():
        return None

    return pd.read_csv(
        HISTORY_PATH
    )


def history_value(
    history,
    epoch,
    column,
):

    if (
        history is None
        or epoch is None
        or column not in history.columns
    ):
        return np.nan

    rows = history[
        history["epoch"]
        == epoch
    ]

    if len(rows) == 0:
        return np.nan

    return float(
        rows.iloc[-1][
            column
        ]
    )


# =========================================================
# Checkpoints
# =========================================================

def get_checkpoints():

    files = sorted(
        CHECKPOINT_DIR.glob(
            "epoch_*.pt"
        )
    )

    best = (
        CHECKPOINT_DIR
        / "best_val_cycle_x.pt"
    )

    if best.exists():
        files.append(
            best
        )

    return files


# =========================================================
# Main
# =========================================================

@torch.no_grad()
def main():

    set_seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # =====================================================
    # Config
    # =====================================================

    config = load_config(
        CONFIG_PATH
    )

    split_csv = (
        PROJECT_ROOT
        / config[
            "data"
        ][
            "split_csv"
        ]
    )

    # =====================================================
    # Validation datasets
    # =====================================================

    val_maf = SingleDomainDataset(
        split_csv=
            split_csv,

        split="val",

        domain="x",

        mode="maf",
    )

    val_he = SingleDomainDataset(
        split_csv=
            split_csv,

        split="val",

        domain="y",

        mode="maf",
    )

    maf_loader = DataLoader(
        val_maf,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
        num_workers=
            NUM_WORKERS,
        pin_memory=True,
    )

    he_loader = DataLoader(
        val_he,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
        num_workers=
            NUM_WORKERS,
        pin_memory=True,
    )

    print()

    print(
        "Validation MAF:",
        len(
            val_maf
        ),
    )

    print(
        "Validation H&E:",
        len(
            val_he
        ),
    )

    # =====================================================
    # History
    # =====================================================

    history = (
        load_history()
    )

    # =====================================================
    # Checkpoints
    # =====================================================

    checkpoints = (
        get_checkpoints()
    )

    print()

    print(
        "Checkpoints:"
    )

    for path in checkpoints:
        print(
            " ",
            path.name,
        )

    results = []

    # =====================================================
    # Evaluate each checkpoint
    # =====================================================

    for checkpoint_path in checkpoints:

        print()

        print(
            "=" * 72
        )

        print(
            "EVALUATING:",
            checkpoint_path.name,
        )

        print(
            "=" * 72
        )

        model, epoch = (
            load_model(
                checkpoint_path,
                config,
                device,
            )
        )

        print(
            "Epoch:",
            epoch,
        )

        # =================================================
        # Metrics
        # =================================================

        fid = (
            FrechetInceptionDistance(
                feature=2048,
                normalize=True,
                reset_real_features=True,
            )
            .set_dtype(
                torch.float64
            )
            .to(
                device
            )
        )

        kid = (
            KernelInceptionDistance(
                feature=2048,
                subsets=
                    KID_SUBSETS,
                subset_size=
                    KID_SUBSET_SIZE,
                normalize=True,
                reset_real_features=True,
            )
            .to(
                device
            )
        )

        # =================================================
        # Real H&E
        # =================================================

        print(
            "Updating real H&E..."
        )

        for batch_idx, batch in enumerate(
            he_loader
        ):

            real_he = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            real_he = (
                to_metric_image(
                    real_he
                )
            )

            fid.update(
                real_he,
                real=True,
            )

            kid.update(
                real_he,
                real=True,
            )

            if (
                batch_idx == 0
                or (batch_idx + 1)
                % 25 == 0
            ):

                print(
                    f"  Real batch "
                    f"{batch_idx + 1}/"
                    f"{len(he_loader)}"
                )

        # =================================================
        # Fake H&E
        # =================================================

        print(
            "Generating fake H&E..."
        )

        for batch_idx, batch in enumerate(
            maf_loader
        ):

            maf = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            fake_he = (
                model.G_XY(
                    maf
                )
            )

            fake_he = (
                to_metric_image(
                    fake_he
                )
            )

            fid.update(
                fake_he,
                real=False,
            )

            kid.update(
                fake_he,
                real=False,
            )

            if (
                batch_idx == 0
                or (batch_idx + 1)
                % 25 == 0
            ):

                print(
                    f"  Fake batch "
                    f"{batch_idx + 1}/"
                    f"{len(maf_loader)}"
                )

        # =================================================
        # Compute
        # =================================================

        print(
            "Computing FID..."
        )

        fid_score = (
            fid.compute()
            .detach()
            .cpu()
            .item()
        )

        print(
            "Computing KID..."
        )

        kid_mean, kid_std = (
            kid.compute()
        )

        kid_mean = (
            kid_mean
            .detach()
            .cpu()
            .item()
        )

        kid_std = (
            kid_std
            .detach()
            .cpu()
            .item()
        )

        print()

        print(
            f"FID:      "
            f"{fid_score:.6f}"
        )

        print(
            f"KID mean: "
            f"{kid_mean:.8f}"
        )

        print(
            f"KID std:  "
            f"{kid_std:.8f}"
        )

        # =================================================
        # Cycle metrics
        # =================================================

        val_cycle_x = (
            history_value(
                history,
                epoch,
                "val_cycle_x",
            )
        )

        val_cycle_y = (
            history_value(
                history,
                epoch,
                "val_cycle_y",
            )
        )

        val_cycle_mean = (
            history_value(
                history,
                epoch,
                "val_cycle_mean",
            )
        )

        results.append(
            {
                "checkpoint":
                    checkpoint_path.name,

                "epoch":
                    epoch,

                "FID":
                    fid_score,

                "KID_mean":
                    kid_mean,

                "KID_std":
                    kid_std,

                "val_cycle_X":
                    val_cycle_x,

                "val_cycle_Y":
                    val_cycle_y,

                "val_cycle_mean":
                    val_cycle_mean,

                "num_fake":
                    len(
                        val_maf
                    ),

                "num_real":
                    len(
                        val_he
                    ),
            }
        )

        # =================================================
        # Cleanup
        # =================================================

        del fid
        del kid
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =====================================================
    # Save
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "epoch",
                "checkpoint",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()

    print(
        "=" * 72
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
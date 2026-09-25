from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import torch
import yaml
from torch.utils.data import DataLoader

from src.models.ecyclegan import ECycleGAN
from src.data.datasets import SingleDomainDataset
from src.utils.visualization import save_translation_preview


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "maf_teacher.yaml"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "maf_teacher"
    / "checkpoints"
    / "epoch_001.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maf_teacher"
    / "previews"
    / "epoch_001_rgb"
)


def main():

    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # =====================================================
    # Config
    # =====================================================

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(
            f
        )

    model_cfg = config[
        "model"
    ]

    data_cfg = config[
        "data"
    ]

    # =====================================================
    # Model
    # =====================================================

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

    # =====================================================
    # Load checkpoint
    # =====================================================

    print(
        "Loading:",
        CHECKPOINT_PATH,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    # Try standard checkpoint key first
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "model" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model"
            ]
        )

    else:

        # In case checkpoint itself is the state dict
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "Checkpoint loaded."
    )

    # =====================================================
    # Validation datasets
    # =====================================================

    split_csv = (
        PROJECT_ROOT
        / data_cfg[
            "split_csv"
        ]
    )

    val_x_dataset = (
        SingleDomainDataset(
            split_csv=
                split_csv,

            split="val",

            domain="x",

            mode="maf",
        )
    )

    val_y_dataset = (
        SingleDomainDataset(
            split_csv=
                split_csv,

            split="val",

            domain="y",

            mode="maf",
        )
    )

    # =====================================================
    # Fixed preview batches
    # =====================================================

    val_x_loader = DataLoader(
        val_x_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    val_y_loader = DataLoader(
        val_y_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    x_batch = next(
        iter(
            val_x_loader
        )
    )

    y_batch = next(
        iter(
            val_y_loader
        )
    )

    # =====================================================
    # Generate previews
    # =====================================================

    save_translation_preview(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        output_dir=
            OUTPUT_DIR,
        device=device,
        max_samples=4,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "RGB PREVIEW REGENERATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
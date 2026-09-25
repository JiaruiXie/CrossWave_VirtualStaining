from pathlib import Path
import argparse
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data.datasets import SingleDomainDataset
from src.models.ecyclegan import ECycleGAN


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize original fake H&E and a manually "
            "contrast-inverted version of fake H&E."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/maf_teacher_stable_b_inverted_aux.yaml",
        help="Experiment YAML config.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/maf_teacher_stable_b_inverted_aux/"
            "checkpoints/best_fid.pt"
        ),
        help="Checkpoint path.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of deterministic validation MAF samples.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/maf_teacher_stable_b_inverted_aux/"
            "fake_y_inversion"
        ),
        help="Directory for saved images.",
    )

    return parser.parse_args()


def resolve_path(path_string):
    path = Path(path_string)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def tensor_to_rgb_uint8(x):
    """
    Convert a single image tensor in [-1, 1] to RGB uint8.

    Supported:
        [1,H,W] -> repeated grayscale
        [2,H,W] -> MAF preview [C0, 0, C2]
        [3,H,W] -> RGB
    """

    x = x.detach().float().cpu()

    if x.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], got shape {tuple(x.shape)}"
        )

    channels = int(x.shape[0])

    if channels == 1:
        x = x.repeat(3, 1, 1)

    elif channels == 2:
        c0 = x[0:1]
        c2 = x[1:2]

        empty = torch.full_like(
            c0,
            -1.0,
        )

        x = torch.cat(
            [c0, empty, c2],
            dim=0,
        )

    elif channels == 3:
        pass

    else:
        raise ValueError(
            "Expected 1-, 2-, or 3-channel image, "
            f"got {channels} channels."
        )

    x = (x + 1.0) / 2.0
    x = torch.clamp(
        x,
        0.0,
        1.0,
    )

    x = (
        x.permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)

    return x


def make_panel(image, title):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            image,
            mode="RGB",
        )

    title_height = 30

    panel = Image.new(
        "RGB",
        (
            image.width,
            image.height + title_height,
        ),
        "white",
    )

    panel.paste(
        image,
        (0, title_height),
    )

    draw = ImageDraw.Draw(panel)

    draw.text(
        (7, 8),
        title,
        fill="black",
    )

    return panel


def save_comparison(
    real_x,
    fake_y,
    fake_y_inverted,
    output_path,
):
    real_x_rgb = tensor_to_rgb_uint8(
        real_x
    )

    fake_y_rgb = tensor_to_rgb_uint8(
        fake_y
    )

    fake_y_inv_rgb = tensor_to_rgb_uint8(
        fake_y_inverted
    )

    panels = [
        make_panel(
            real_x_rgb,
            "Real MAF [C0, 0, C2]",
        ),
        make_panel(
            fake_y_rgb,
            "Generated H&E",
        ),
        make_panel(
            fake_y_inv_rgb,
            "Generated H&E - manually inverted",
        ),
    ]

    width = sum(
        panel.width
        for panel in panels
    )

    height = max(
        panel.height
        for panel in panels
    )

    montage = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    x_offset = 0

    for panel in panels:
        montage.paste(
            panel,
            (x_offset, 0),
        )
        x_offset += panel.width

    montage.save(
        output_path
    )


def build_model(config, device):
    model_cfg = config["model"]

    model = ECycleGAN(
        x_channels=int(
            model_cfg["x_channels"]
        ),
        y_channels=int(
            model_cfg["y_channels"]
        ),
        base_channels=int(
            model_cfg.get(
                "base_channels",
                64,
            )
        ),
        num_res_blocks=int(
            model_cfg.get(
                "num_res_blocks",
                9,
            )
        ),
        padding_type=model_cfg.get(
            "padding_type",
            "reflect",
        ),
    ).to(device)

    return model


def load_model_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    epoch = (
        checkpoint.get("epoch")
        if isinstance(checkpoint, dict)
        else None
    )

    best_fid = (
        checkpoint.get("best_fid")
        if isinstance(checkpoint, dict)
        else None
    )

    return epoch, best_fid


@torch.no_grad()
def main():
    args = parse_args()

    config_path = resolve_path(
        args.config
    )

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
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

    print(
        "Config:",
        config_path,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    model = build_model(
        config,
        device,
    )

    epoch, best_fid = (
        load_model_checkpoint(
            model,
            checkpoint_path,
            device,
        )
    )

    print(
        "Checkpoint epoch:",
        epoch,
    )

    print(
        "Checkpoint best FID:",
        best_fid,
    )

    data_cfg = config["data"]

    split_csv = resolve_path(
        data_cfg["split_csv"]
    )

    dataset = SingleDomainDataset(
        split_csv=split_csv,
        split="val",
        domain="x",
        mode=data_cfg.get(
            "mode",
            "maf",
        ),
    )

    print(
        "Validation MAF images:",
        len(dataset),
    )

    num_samples = min(
        int(args.num_samples),
        len(dataset),
    )

    if num_samples <= 0:
        raise ValueError(
            "--num-samples must be >= 1."
        )

    if num_samples == 1:
        indices = [0]
    else:
        indices = [
            round(
                i
                * (len(dataset) - 1)
                / (num_samples - 1)
            )
            for i in range(
                num_samples
            )
        ]

    print(
        "Selected indices:",
        indices,
    )

    for sample_number, index in enumerate(
        indices
    ):
        batch = dataset[index]

        real_x = batch["image"].unsqueeze(
            0
        ).to(
            device
        )

        fake_y = model.G_XY(
            real_x
        )

        # Diagnostic only:
        # fake_y is in [-1, 1], so global RGB inversion is -fake_y.
        fake_y_inverted = -fake_y

        real_x_single = real_x[0]
        fake_y_single = fake_y[0]
        fake_y_inv_single = (
            fake_y_inverted[0]
        )

        Image.fromarray(
            tensor_to_rgb_uint8(
                real_x_single
            ),
            mode="RGB",
        ).save(
            output_dir
            / (
                f"sample_{sample_number:02d}"
                "_real_x.tif"
            )
        )

        Image.fromarray(
            tensor_to_rgb_uint8(
                fake_y_single
            ),
            mode="RGB",
        ).save(
            output_dir
            / (
                f"sample_{sample_number:02d}"
                "_fake_y.tif"
            )
        )

        Image.fromarray(
            tensor_to_rgb_uint8(
                fake_y_inv_single
            ),
            mode="RGB",
        ).save(
            output_dir
            / (
                f"sample_{sample_number:02d}"
                "_fake_y_inverted.tif"
            )
        )

        comparison_path = (
            output_dir
            / (
                f"sample_{sample_number:02d}"
                "_comparison.png"
            )
        )

        save_comparison(
            real_x=
                real_x_single,

            fake_y=
                fake_y_single,

            fake_y_inverted=
                fake_y_inv_single,

            output_path=
                comparison_path,
        )

        print(
            f"Sample {sample_number:02d} "
            f"| dataset index {index} "
            f"| saved {comparison_path.name}"
        )

    print()
    print(
        "Done."
    )

    print(
        "Output directory:",
        output_dir,
    )

    print()
    print(
        "IMPORTANT: fake_y_inverted is only a diagnostic "
        "visualization. It is not used for training."
    )


if __name__ == "__main__":
    main()

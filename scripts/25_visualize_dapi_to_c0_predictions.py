from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps, ImageDraw, ImageFont


# ============================================================
# 1. MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.block(
            x
        )


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ):
        super().__init__()

        b = base_channels

        self.enc1 = DoubleConv(
            in_channels,
            b,
        )

        self.enc2 = DoubleConv(
            b,
            b * 2,
        )

        self.enc3 = DoubleConv(
            b * 2,
            b * 4,
        )

        self.enc4 = DoubleConv(
            b * 4,
            b * 8,
        )

        self.bottleneck = DoubleConv(
            b * 8,
            b * 16,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.up4 = nn.ConvTranspose2d(
            b * 16,
            b * 8,
            kernel_size=2,
            stride=2,
        )

        self.dec4 = DoubleConv(
            b * 16,
            b * 8,
        )

        self.up3 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            b * 8,
            b * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            b * 4,
            b * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            b * 2,
            b,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            b * 2,
            b,
        )

        self.out = nn.Conv2d(
            b,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x,
    ):

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        bottleneck = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(
            bottleneck
        )

        d4 = torch.cat(
            [d4, e4],
            dim=1,
        )

        d4 = self.dec4(
            d4
        )

        d3 = self.up3(
            d4
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )

        d3 = self.dec3(
            d3
        )

        d2 = self.up2(
            d3
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.dec2(
            d2
        )

        d1 = self.up1(
            d2
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.dec1(
            d1
        )

        out = self.out(
            d1
        )

        return torch.sigmoid(
            out
        )


# ============================================================
# 2. HELPERS
# ============================================================

def read_dapi(
    path,
):

    image = tifffile.imread(
        path
    )

    if image.ndim == 3:

        if image.shape[-1] == 1:

            image = image[
                ...,
                0,
            ]

        else:

            raise ValueError(
                f"Unexpected DAPI shape: "
                f"{image.shape} "
                f"for {path}"
            )

    if image.ndim != 2:

        raise ValueError(
            f"DAPI must be 2D. "
            f"Got {image.shape} "
            f"for {path}"
        )

    image = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    return image


def read_c0(
    path,
):

    maf = tifffile.imread(
        path
    )

    if (
        maf.ndim != 3
        or maf.shape[-1] < 1
    ):

        raise ValueError(
            f"Unexpected MAF shape: "
            f"{maf.shape} "
            f"for {path}"
        )

    c0 = maf[
        ...,
        0,
    ]

    c0 = (
        c0.astype(
            np.float32
        )
        / 255.0
    )

    return c0


def to_uint8(
    image,
):

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = (
        image
        * 255.0
    ).round().astype(
        np.uint8
    )

    return image


def save_grayscale(
    image,
    output_path,
):

    image_uint8 = to_uint8(
        image
    )

    Image.fromarray(
        image_uint8,
        mode="L",
    ).save(
        output_path
    )


# ============================================================
# 3. CREATE SIDE-BY-SIDE PANEL
# ============================================================

def make_comparison_panel(
    dapi,
    predicted_c0,
    ground_truth_c0,
    output_path,
):

    dapi_img = Image.fromarray(
        to_uint8(
            dapi
        ),
        mode="L",
    )

    pred_img = Image.fromarray(
        to_uint8(
            predicted_c0
        ),
        mode="L",
    )

    gt_img = Image.fromarray(
        to_uint8(
            ground_truth_c0
        ),
        mode="L",
    )

    width, height = (
        dapi_img.size
    )

    label_height = 34

    canvas = Image.new(
        "L",
        (
            width * 3,
            height + label_height,
        ),
        color=255,
    )

    canvas.paste(
        dapi_img,
        (
            0,
            label_height,
        ),
    )

    canvas.paste(
        pred_img,
        (
            width,
            label_height,
        ),
    )

    canvas.paste(
        gt_img,
        (
            width * 2,
            label_height,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    labels = [
        "DAPI",
        "Predicted C0",
        "Ground-truth C0",
    ]

    for i, label in enumerate(
        labels
    ):

        bbox = draw.textbbox(
            (0, 0),
            label,
        )

        text_width = (
            bbox[2]
            - bbox[0]
        )

        x = (
            i * width
            + (
                width
                - text_width
            )
            / 2
        )

        draw.text(
            (
                x,
                10,
            ),
            label,
            fill=0,
        )

    canvas.save(
        output_path
    )


# ============================================================
# 4. LOAD CHECKPOINT
# ============================================================

def load_model(
    checkpoint_path,
    device,
    base_channels=32,
):

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
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

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Loaded checkpoint:",
        checkpoint_path,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
    ):

        if "epoch" in checkpoint:

            print(
                "Checkpoint epoch:",
                checkpoint["epoch"],
            )

        if "val_ssim" in checkpoint:

            print(
                "Checkpoint val SSIM:",
                checkpoint["val_ssim"],
            )

    return model


# ============================================================
# 5. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-csv",
        type=str,
        default=(
            "data/metadata/"
            "data_split.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/"
            "redundancy_dapi_to_c0/"
            "unet_dapi_to_c0_v1/"
            "checkpoints/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/"
            "redundancy_dapi_to_c0/"
            "unet_dapi_to_c0_v1/"
            "visual_examples"
        ),
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    split_csv = (
        project_root
        / args.split_csv
    ).resolve()

    checkpoint_path = (
        project_root
        / args.checkpoint
    ).resolve()

    output_dir = (
        project_root
        / args.output_dir
    ).resolve()

    panels_dir = (
        output_dir
        / "comparison_panels"
    )

    individual_dir = (
        output_dir
        / "individual"
    )

    panels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    individual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "DAPI -> C0 VISUALIZATION"
    )

    print(
        "=" * 80
    )

    print(
        "Device:",
        device,
    )

    print(
        "Split CSV:",
        split_csv,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Output:",
        output_dir,
    )


    # ========================================================
    # 6. LOAD DATAFRAME
    # ========================================================

    df = pd.read_csv(
        split_csv
    )

    subset = (
        df[
            (
                df["domain"]
                == "maf"
            )
            &
            (
                df["split"]
                == args.split
            )
        ]
        .reset_index(
            drop=True
        )
    )

    if len(subset) == 0:

        raise RuntimeError(
            f"No paired MAF/DAPI rows "
            f"found for split="
            f"{args.split}"
        )

    print(
        f"Available {args.split} "
        f"samples: "
        f"{len(subset):,}"
    )


    # ========================================================
    # 7. RANDOM SAMPLE
    # ========================================================

    num_images = min(
        args.num_images,
        len(subset),
    )

    rng = np.random.default_rng(
        args.seed
    )

    selected_indices = rng.choice(
        len(subset),
        size=num_images,
        replace=False,
    )

    selected_indices = sorted(
        selected_indices.tolist()
    )


    # ========================================================
    # 8. LOAD MODEL
    # ========================================================

    model = load_model(
        checkpoint_path=
            checkpoint_path,
        device=
            device,
        base_channels=
            args.base_channels,
    )


    # ========================================================
    # 9. GENERATE EXAMPLES
    # ========================================================

    records = []

    for example_number, index in enumerate(
        selected_indices,
        start=1,
    ):

        row = subset.iloc[
            index
        ]

        dapi_path = Path(
            row["dapi_path"]
        )

        maf_path = Path(
            row["maf_path"]
        )

        filename = str(
            row.get(
                "filename",
                f"sample_{index:05d}",
            )
        )

        dapi = read_dapi(
            dapi_path
        )

        gt_c0 = read_c0(
            maf_path
        )

        if (
            dapi.shape
            != gt_c0.shape
        ):

            raise ValueError(
                f"Shape mismatch for "
                f"{filename}: "
                f"DAPI={dapi.shape}, "
                f"C0={gt_c0.shape}"
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        input_tensor = (
            torch.from_numpy(
                dapi
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.no_grad():

            prediction = model(
                input_tensor
            )

        predicted_c0 = (
            prediction[
                0,
                0,
            ]
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Safe file stem
        # ----------------------------------------------------

        stem = Path(
            filename
        ).stem

        prefix = (
            f"{example_number:02d}_"
            f"{stem}"
        )


        # ----------------------------------------------------
        # Individual images
        # ----------------------------------------------------

        dapi_out = (
            individual_dir
            / f"{prefix}_DAPI.png"
        )

        pred_out = (
            individual_dir
            / f"{prefix}_Predicted_C0.png"
        )

        gt_out = (
            individual_dir
            / f"{prefix}_GroundTruth_C0.png"
        )

        save_grayscale(
            dapi,
            dapi_out,
        )

        save_grayscale(
            predicted_c0,
            pred_out,
        )

        save_grayscale(
            gt_c0,
            gt_out,
        )


        # ----------------------------------------------------
        # Comparison panel
        # ----------------------------------------------------

        panel_out = (
            panels_dir
            / f"{prefix}_comparison.png"
        )

        make_comparison_panel(
            dapi=
                dapi,
            predicted_c0=
                predicted_c0,
            ground_truth_c0=
                gt_c0,
            output_path=
                panel_out,
        )


        # ----------------------------------------------------
        # Simple per-image metrics
        # ----------------------------------------------------

        error = (
            predicted_c0
            - gt_c0
        )

        mae = float(
            np.mean(
                np.abs(
                    error
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        pred_flat = (
            predicted_c0
            .reshape(-1)
        )

        gt_flat = (
            gt_c0
            .reshape(-1)
        )

        pred_centered = (
            pred_flat
            - pred_flat.mean()
        )

        gt_centered = (
            gt_flat
            - gt_flat.mean()
        )

        denominator = (
            np.sqrt(
                np.sum(
                    pred_centered ** 2
                )
            )
            *
            np.sqrt(
                np.sum(
                    gt_centered ** 2
                )
            )
        )

        if denominator > 0:

            pearson = float(
                np.sum(
                    pred_centered
                    * gt_centered
                )
                / denominator
            )

        else:

            pearson = np.nan


        records.append(
            {
                "example":
                    example_number,
                "dataset_index":
                    index,
                "filename":
                    filename,
                "dapi_path":
                    str(dapi_path),
                "maf_path":
                    str(maf_path),
                "mae":
                    mae,
                "rmse":
                    rmse,
                "pearson":
                    pearson,
                "comparison_panel":
                    str(panel_out),
            }
        )

        print(
            f"[{example_number:02d}/"
            f"{num_images:02d}] "
            f"{filename} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"Pearson={pearson:.4f}"
        )


    # ========================================================
    # 10. SAVE MANIFEST
    # ========================================================

    manifest = pd.DataFrame(
        records
    )

    manifest_path = (
        output_dir
        / "visual_examples_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )


    print()
    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print(
        f"Saved "
        f"{num_images} examples."
    )

    print(
        "Comparison panels:"
    )

    print(
        panels_dir
    )

    print(
        "Individual images:"
    )

    print(
        individual_dir
    )

    print(
        "Manifest:"
    )

    print(
        manifest_path
    )


if __name__ == "__main__":
    main()
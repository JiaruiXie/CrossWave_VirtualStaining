from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]

OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "redundancy_dapi_to_c0"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed=42,
):

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

    torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET
# ============================================================

class DAPIToC0Dataset(
    Dataset
):

    """
    Paired DAPI -> C0 reconstruction dataset.

    The existing data_split.csv contains rows where:

        domain == "maf"

    and each such row contains:

        maf_path
        dapi_path

    The released MAF TIFF contains:

        C0 = channel 0
        C1 = empty
        C2 = channel 2

    Therefore:

        input  = DAPI
        target = MAF[..., 0]
    """

    def __init__(
        self,
        split_csv,
        split,
        augment=False,
    ):

        super().__init__()

        self.split = split
        self.augment = augment

        df = pd.read_csv(
            split_csv
        )

        self.df = (
            df[
                (df["domain"] == "maf")
                &
                (df["split"] == split)
            ]
            .reset_index(
                drop=True
            )
        )

        if len(
            self.df
        ) == 0:

            raise RuntimeError(
                f"No paired MAF/DAPI rows found "
                f"for split={split}"
            )

        print(
            f"DAPIToC0Dataset | "
            f"split={split}"
        )

        print(
            f"  samples: "
            f"{len(self.df):,}"
        )

    def __len__(
        self
    ):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        dapi_path = Path(
            row[
                "dapi_path"
            ]
        )

        maf_path = Path(
            row[
                "maf_path"
            ]
        )

        # ====================================================
        # LOAD DAPI
        # ====================================================

        dapi = tifffile.imread(
            dapi_path
        )

        if dapi.ndim == 3:

            if dapi.shape[-1] == 1:

                dapi = dapi[
                    ...,
                    0
                ]

            else:

                raise ValueError(
                    f"Unexpected DAPI shape: "
                    f"{dapi.shape}\n"
                    f"File: {dapi_path}"
                )

        # ====================================================
        # LOAD MAF
        # ====================================================

        maf = tifffile.imread(
            maf_path
        )

        if (
            maf.ndim != 3
            or maf.shape[-1] < 1
        ):

            raise ValueError(
                f"Unexpected MAF shape: "
                f"{maf.shape}\n"
                f"File: {maf_path}"
            )

        # ----------------------------------------------------
        # Target = C0
        # ----------------------------------------------------

        c0 = maf[
            ...,
            0
        ]

        # ====================================================
        # SHAPE CHECK
        # ====================================================

        if dapi.shape != c0.shape:

            raise ValueError(
                "DAPI/C0 shape mismatch:\n"
                f"  DAPI: {dapi.shape}\n"
                f"  C0:   {c0.shape}\n"
                f"  DAPI path: {dapi_path}\n"
                f"  MAF path:  {maf_path}"
            )

        # ====================================================
        # NORMALIZATION
        #
        # Original dataset is uint8:
        #
        # [0, 255] -> [0, 1]
        # ====================================================

        dapi = (
            dapi.astype(
                np.float32
            )
            / 255.0
        )

        c0 = (
            c0.astype(
                np.float32
            )
            / 255.0
        )

        # ====================================================
        # PAIRED SPATIAL AUGMENTATION
        # ====================================================

        if (
            self.augment
            and self.split == "train"
        ):

            # ------------------------------------------------
            # Horizontal flip
            # ------------------------------------------------

            if np.random.rand() < 0.5:

                dapi = np.flip(
                    dapi,
                    axis=1,
                ).copy()

                c0 = np.flip(
                    c0,
                    axis=1,
                ).copy()

            # ------------------------------------------------
            # Vertical flip
            # ------------------------------------------------

            if np.random.rand() < 0.5:

                dapi = np.flip(
                    dapi,
                    axis=0,
                ).copy()

                c0 = np.flip(
                    c0,
                    axis=0,
                ).copy()

            # ------------------------------------------------
            # 0 / 90 / 180 / 270 degree rotation
            # ------------------------------------------------

            k = np.random.randint(
                0,
                4,
            )

            if k > 0:

                dapi = np.rot90(
                    dapi,
                    k=k,
                    axes=(0, 1),
                ).copy()

                c0 = np.rot90(
                    c0,
                    k=k,
                    axes=(0, 1),
                ).copy()

        # ====================================================
        # H x W -> 1 x H x W
        # ====================================================

        dapi = torch.from_numpy(
            dapi[
                None,
                ...
            ]
        ).float()

        c0 = torch.from_numpy(
            c0[
                None,
                ...
            ]
        ).float()

        return {
            "dapi":
                dapi,

            "c0":
                c0,

            "filename":
                str(
                    row[
                        "filename"
                    ]
                ),
        }


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(
    nn.Module
):

    """
    Two-convolution residual block.

    BatchNorm is used consistently with the current
    CrossWave reconstruction experiments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=
                in_channels,

            out_channels=
                out_channels,

            kernel_size=
                3,

            stride=
                stride,

            padding=
                1,

            bias=
                False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            in_channels=
                out_channels,

            out_channels=
                out_channels,

            kernel_size=
                3,

            stride=
                1,

            padding=
                1,

            bias=
                False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        # ----------------------------------------------------
        # Residual projection when shape changes
        # ----------------------------------------------------

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels=
                        in_channels,

                    out_channels=
                        out_channels,

                    kernel_size=
                        1,

                    stride=
                        stride,

                    bias=
                        False,
                ),

                nn.BatchNorm2d(
                    out_channels
                ),
            )

        else:

            self.shortcut = nn.Identity()

    def forward(
        self,
        x,
    ):

        identity = self.shortcut(
            x
        )

        out = self.conv1(
            x
        )

        out = self.bn1(
            out
        )

        out = self.relu(
            out
        )

        out = self.conv2(
            out
        )

        out = self.bn2(
            out
        )

        out = (
            out
            + identity
        )

        out = self.relu(
            out
        )

        return out


# ============================================================
# RESUNET
# ============================================================

class ResUNet(
    nn.Module
):

    def __init__(
        self,
        base_channels=32,
    ):

        super().__init__()

        b = int(
            base_channels
        )

        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = ResidualBlock(
            in_channels=
                1,
            out_channels=
                b,
            stride=
                1,
        )

        self.enc2 = ResidualBlock(
            in_channels=
                b,
            out_channels=
                b * 2,
            stride=
                2,
        )

        self.enc3 = ResidualBlock(
            in_channels=
                b * 2,
            out_channels=
                b * 4,
            stride=
                2,
        )

        self.enc4 = ResidualBlock(
            in_channels=
                b * 4,
            out_channels=
                b * 8,
            stride=
                2,
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = ResidualBlock(
            in_channels=
                b * 8,
            out_channels=
                b * 16,
            stride=
                2,
        )

        # ====================================================
        # DECODER 4
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            in_channels=
                b * 16,

            out_channels=
                b * 8,

            kernel_size=
                2,

            stride=
                2,
        )

        self.dec4 = ResidualBlock(
            in_channels=
                b * 16,

            out_channels=
                b * 8,
        )

        # ====================================================
        # DECODER 3
        # ====================================================

        self.up3 = nn.ConvTranspose2d(
            in_channels=
                b * 8,

            out_channels=
                b * 4,

            kernel_size=
                2,

            stride=
                2,
        )

        self.dec3 = ResidualBlock(
            in_channels=
                b * 8,

            out_channels=
                b * 4,
        )

        # ====================================================
        # DECODER 2
        # ====================================================

        self.up2 = nn.ConvTranspose2d(
            in_channels=
                b * 4,

            out_channels=
                b * 2,

            kernel_size=
                2,

            stride=
                2,
        )

        self.dec2 = ResidualBlock(
            in_channels=
                b * 4,

            out_channels=
                b * 2,
        )

        # ====================================================
        # DECODER 1
        # ====================================================

        self.up1 = nn.ConvTranspose2d(
            in_channels=
                b * 2,

            out_channels=
                b,

            kernel_size=
                2,

            stride=
                2,
        )

        self.dec1 = ResidualBlock(
            in_channels=
                b * 2,

            out_channels=
                b,
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.output = nn.Conv2d(
            in_channels=
                b,

            out_channels=
                1,

            kernel_size=
                1,
        )

    def forward(
        self,
        x,
    ):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            e1
        )

        e3 = self.enc3(
            e2
        )

        e4 = self.enc4(
            e3
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        bottleneck = self.bottleneck(
            e4
        )

        # ====================================================
        # DECODER 4
        # ====================================================

        d4 = self.up4(
            bottleneck
        )

        d4 = torch.cat(
            [
                d4,
                e4,
            ],
            dim=1,
        )

        d4 = self.dec4(
            d4
        )

        # ====================================================
        # DECODER 3
        # ====================================================

        d3 = self.up3(
            d4
        )

        d3 = torch.cat(
            [
                d3,
                e3,
            ],
            dim=1,
        )

        d3 = self.dec3(
            d3
        )

        # ====================================================
        # DECODER 2
        # ====================================================

        d2 = self.up2(
            d3
        )

        d2 = torch.cat(
            [
                d2,
                e2,
            ],
            dim=1,
        )

        d2 = self.dec2(
            d2
        )

        # ====================================================
        # DECODER 1
        # ====================================================

        d1 = self.up1(
            d2
        )

        d1 = torch.cat(
            [
                d1,
                e1,
            ],
            dim=1,
        )

        d1 = self.dec1(
            d1
        )

        # ====================================================
        # OUTPUT [0,1]
        # ====================================================

        output = torch.sigmoid(
            self.output(
                d1
            )
        )

        return output


# ============================================================
# SSIM
# ============================================================

def ssim_torch(
    x,
    y,
    window_size=11,
):

    """
    Simple differentiable SSIM for [0,1] grayscale images.
    """

    c1 = (
        0.01 ** 2
    )

    c2 = (
        0.03 ** 2
    )

    padding = (
        window_size
        // 2
    )

    mu_x = F.avg_pool2d(
        x,
        kernel_size=
            window_size,
        stride=
            1,
        padding=
            padding,
    )

    mu_y = F.avg_pool2d(
        y,
        kernel_size=
            window_size,
        stride=
            1,
        padding=
            padding,
    )

    sigma_x = (
        F.avg_pool2d(
            x * x,
            kernel_size=
                window_size,
            stride=
                1,
            padding=
                padding,
        )
        - mu_x * mu_x
    )

    sigma_y = (
        F.avg_pool2d(
            y * y,
            kernel_size=
                window_size,
            stride=
                1,
            padding=
                padding,
        )
        - mu_y * mu_y
    )

    sigma_xy = (
        F.avg_pool2d(
            x * y,
            kernel_size=
                window_size,
            stride=
                1,
            padding=
                padding,
        )
        - mu_x * mu_y
    )

    numerator = (
        (
            2.0
            * mu_x
            * mu_y
            + c1
        )
        *
        (
            2.0
            * sigma_xy
            + c2
        )
    )

    denominator = (
        (
            mu_x ** 2
            + mu_y ** 2
            + c1
        )
        *
        (
            sigma_x
            + sigma_y
            + c2
        )
    )

    ssim_map = (
        numerator
        /
        (
            denominator
            + 1e-8
        )
    )

    return ssim_map.mean()


# ============================================================
# LOSS
# ============================================================

def reconstruction_loss(
    prediction,
    target,
    lambda_l1,
    lambda_ssim,
):

    l1 = F.l1_loss(
        prediction,
        target,
    )

    ssim = ssim_torch(
        prediction,
        target,
    )

    ssim_loss = (
        1.0
        - ssim
    )

    loss = (
        lambda_l1
        * l1
        +
        lambda_ssim
        * ssim_loss
    )

    return (
        loss,
        l1,
        ssim,
    )


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    lambda_l1,
    lambda_ssim,
    optimizer=None,
):

    training = (
        optimizer
        is not None
    )

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0

    total_samples = 0

    for batch in loader:

        dapi = batch[
            "dapi"
        ].to(
            device,
            non_blocking=True,
        )

        c0 = batch[
            "c0"
        ].to(
            device,
            non_blocking=True,
        )

        batch_size = int(
            dapi.shape[
                0
            ]
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            prediction = model(
                dapi
            )

            (
                loss,
                l1,
                ssim,
            ) = reconstruction_loss(
                prediction=
                    prediction,

                target=
                    c0,

                lambda_l1=
                    lambda_l1,

                lambda_ssim=
                    lambda_ssim,
            )

            if training:

                loss.backward()

                optimizer.step()

        total_loss += (
            float(
                loss
                .detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_l1 += (
            float(
                l1
                .detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_ssim += (
            float(
                ssim
                .detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_samples += (
            batch_size
        )

    denominator = max(
        total_samples,
        1,
    )

    return {
        "loss":
            total_loss
            / denominator,

        "l1":
            total_l1
            / denominator,

        "ssim":
            total_ssim
            / denominator,
    }


# ============================================================
# TEST METRICS
# ============================================================

def calculate_pearson(
    prediction,
    target,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    ).reshape(
        -1
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    ).reshape(
        -1
    )

    prediction_centered = (
        prediction
        - prediction.mean()
    )

    target_centered = (
        target
        - target.mean()
    )

    numerator = float(
        np.sum(
            prediction_centered
            * target_centered
        )
    )

    denominator = math.sqrt(

        float(
            np.sum(
                prediction_centered
                ** 2
            )
        )

        *

        float(
            np.sum(
                target_centered
                ** 2
            )
        )
    )

    if denominator <= 1e-12:

        return np.nan

    return float(
        numerator
        / denominator
    )


@torch.no_grad()
def evaluate_test(
    model,
    loader,
    device,
    output_dir,
):

    model.eval()

    rows = []

    for batch in loader:

        dapi = batch[
            "dapi"
        ].to(
            device,
            non_blocking=True,
        )

        c0 = batch[
            "c0"
        ].to(
            device,
            non_blocking=True,
        )

        prediction = model(
            dapi
        )

        batch_size = int(
            prediction.shape[
                0
            ]
        )

        for i in range(
            batch_size
        ):

            pred = (
                prediction[
                    i,
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )

            target = (
                c0[
                    i,
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )

            # =================================================
            # MAE
            # =================================================

            difference = (
                pred
                - target
            )

            mae = float(
                np.mean(
                    np.abs(
                        difference
                    )
                )
            )

            # =================================================
            # RMSE
            # =================================================

            mse = float(
                np.mean(
                    difference
                    ** 2
                )
            )

            rmse = float(
                np.sqrt(
                    mse
                )
            )

            # =================================================
            # PSNR
            # =================================================

            if mse <= 1e-12:

                psnr = 120.0

            else:

                psnr = float(
                    10.0
                    * math.log10(
                        1.0
                        / mse
                    )
                )

            # =================================================
            # SSIM
            # =================================================

            pred_tensor = (
                torch.from_numpy(
                    pred
                )
                [
                    None,
                    None,
                    ...
                ]
                .float()
            )

            target_tensor = (
                torch.from_numpy(
                    target
                )
                [
                    None,
                    None,
                    ...
                ]
                .float()
            )

            ssim = float(
                ssim_torch(
                    pred_tensor,
                    target_tensor,
                )
                .cpu()
                .item()
            )

            # =================================================
            # PEARSON
            # =================================================

            pearson = calculate_pearson(
                pred,
                target,
            )

            rows.append(
                {
                    "filename":
                        batch[
                            "filename"
                        ][
                            i
                        ],

                    "mae":
                        mae,

                    "rmse":
                        rmse,

                    "psnr":
                        psnr,

                    "ssim":
                        ssim,

                    "pearson":
                        pearson,
                }
            )

    metrics_df = pd.DataFrame(
        rows
    )

    per_image_path = (
        output_dir
        / "test_metrics_per_image.csv"
    )

    metrics_df.to_csv(
        per_image_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "n":
            int(
                len(
                    metrics_df
                )
            )
    }

    for metric in [
        "mae",
        "rmse",
        "psnr",
        "ssim",
        "pearson",
    ]:

        values = pd.to_numeric(
            metrics_df[
                metric
            ],
            errors="coerce",
        )

        summary[
            f"{metric}_mean"
        ] = float(
            values.mean()
        )

        summary[
            f"{metric}_std"
        ] = float(
            values.std(
                ddof=1
            )
        )

        summary[
            f"{metric}_median"
        ] = float(
            values.median()
        )

    summary_path = (
        output_dir
        / "test_metrics_summary.csv"
    )

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        summary_path,
        index=False,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a ResUNet to reconstruct "
            "MAF C0 from paired DAPI images."
        )
    )

    # ========================================================
    # DATA
    # ========================================================

    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-ssim",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=
            "resunet_dapi_to_c0_v1",
    )

    args = parser.parse_args()

    # ========================================================
    # RESOLVE SPLIT CSV
    # ========================================================

    split_csv = (
        args.split_csv
    )

    if not split_csv.is_absolute():

        split_csv = (
            ROOT
            / split_csv
        )

    if not split_csv.is_file():

        raise FileNotFoundError(
            "Split CSV not found:\n"
            f"{split_csv}"
        )

    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        args.seed
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        / args.output_name
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "DAPI -> C0 RESUNET "
        "CROSS-CHANNEL RECONSTRUCTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: "
        f"{device}"
    )

    print(
        f"Split CSV: "
        f"{split_csv}"
    )

    print(
        f"Base channels: "
        f"{args.base_channels}"
    )

    print(
        f"Loss: "
        f"{args.lambda_l1} * L1 "
        f"+ {args.lambda_ssim} * (1-SSIM)"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = DAPIToC0Dataset(
        split_csv=
            split_csv,

        split=
            "train",

        augment=
            True,
    )

    val_dataset = DAPIToC0Dataset(
        split_csv=
            split_csv,

        split=
            "val",

        augment=
            False,
    )

    # ========================================================
    # TEST DATASET
    #
    # Your current CSV has no test split.
    #
    # Therefore:
    # - try test
    # - temporarily fall back to val
    #
    # This is acceptable for model development but should
    # NOT be described as an independent test set in the
    # final manuscript.
    # ========================================================

    try:

        test_dataset = DAPIToC0Dataset(
            split_csv=
                split_csv,

            split=
                "test",

            augment=
                False,
        )

        test_split_name = (
            "test"
        )

    except RuntimeError:

        print()
        print(
            "WARNING: no test split found."
        )

        print(
            "Using validation split as "
            "test set for this run."
        )

        test_dataset = DAPIToC0Dataset(
            split_csv=
                split_csv,

            split=
                "val",

            augment=
                False,
        )

        test_split_name = (
            "val"
        )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    loader_kwargs = {
        "batch_size":
            args.batch_size,

        "num_workers":
            args.num_workers,

        "pin_memory":
            torch.cuda.is_available(),

        "drop_last":
            False,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = ResUNet(
        base_channels=
            args.base_channels
    ).to(
        device
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=
            args.lr,

        weight_decay=
            args.weight_decay,
    )

    # ========================================================
    # LR SCHEDULER
    #
    # Keep the same behavior as the existing U-Net experiment.
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,

            mode=
                "min",

            factor=
                0.5,

            patience=
                8,

            min_lr=
                1e-6,
        )
    )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_val_ssim = float(
        "-inf"
    )

    best_val_loss = float(
        "inf"
    )

    best_val_l1 = float(
        "inf"
    )

    best_epoch = -1

    history = []

    best_checkpoint = (
        checkpoint_dir
        / "best_model.pt"
    )

    run_start = time.time()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = (
            time.time()
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_result = run_epoch(
            model=
                model,

            loader=
                train_loader,

            device=
                device,

            lambda_l1=
                args.lambda_l1,

            lambda_ssim=
                args.lambda_ssim,

            optimizer=
                optimizer,
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        val_result = run_epoch(
            model=
                model,

            loader=
                val_loader,

            device=
                device,

            lambda_l1=
                args.lambda_l1,

            lambda_ssim=
                args.lambda_ssim,

            optimizer=
                None,
        )

        # ----------------------------------------------------
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_result[
                "loss"
            ]
        )

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        # ====================================================
        # BEST MODEL:
        # HIGHEST VALIDATION SSIM
        # ====================================================

        improved = (
            val_result[
                "ssim"
            ]
            > best_val_ssim
        )

        if improved:

            best_val_ssim = float(
                val_result[
                    "ssim"
                ]
            )

            best_val_loss = float(
                val_result[
                    "loss"
                ]
            )

            best_val_l1 = float(
                val_result[
                    "l1"
                ]
            )

            best_epoch = int(
                epoch
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_name":
                        "ResUNet",

                    "base_channels":
                        args.base_channels,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_ssim":
                        best_val_ssim,

                    "val_loss":
                        best_val_loss,

                    "val_l1":
                        best_val_l1,

                    "args":
                        vars(
                            args
                        ),
                },
                best_checkpoint,
            )

        # ====================================================
        # HISTORY
        # ====================================================

        history.append(
            {
                "epoch":
                    epoch,

                "lr":
                    current_lr,

                "train_loss":
                    train_result[
                        "loss"
                    ],

                "train_l1":
                    train_result[
                        "l1"
                    ],

                "train_ssim":
                    train_result[
                        "ssim"
                    ],

                "val_loss":
                    val_result[
                        "loss"
                    ],

                "val_l1":
                    val_result[
                        "l1"
                    ],

                "val_ssim":
                    val_result[
                        "ssim"
                    ],

                "best_val_ssim_so_far":
                    best_val_ssim,

                "is_best_ssim":
                    int(
                        improved
                    ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            output_dir
            / "training_history.csv",

            index=False,
        )

        # ====================================================
        # LOGGING
        # ====================================================

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss="
            f"{train_result['loss']:.5f} "
            f"L1="
            f"{train_result['l1']:.5f} "
            f"SSIM="
            f"{train_result['ssim']:.4f} | "
            f"val loss="
            f"{val_result['loss']:.5f} "
            f"L1="
            f"{val_result['l1']:.5f} "
            f"SSIM="
            f"{val_result['ssim']:.4f} | "
            f"lr="
            f"{current_lr:.2e} | "
            f"{elapsed:.1f}s"
            +
            (
                "  <-- BEST SSIM"
                if improved
                else ""
            ),
            flush=True,
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if not best_checkpoint.is_file():

        raise RuntimeError(
            "Best checkpoint was not created."
        )

    checkpoint = torch.load(
        best_checkpoint,

        map_location=
            device,

        weights_only=
            False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()
    print(
        "=" * 80
    )

    print(
        "BEST VALIDATION MODEL"
    )

    print(
        "=" * 80
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best val SSIM: "
        f"{best_val_ssim:.6f}"
    )

    print(
        f"Val L1 at best SSIM epoch: "
        f"{best_val_l1:.6f}"
    )

    print(
        f"Val loss at best SSIM epoch: "
        f"{best_val_loss:.6f}"
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print()
    print(
        f"Evaluating on "
        f"{test_split_name!r} split..."
    )

    summary = evaluate_test(
        model=
            model,

        loader=
            test_loader,

        device=
            device,

        output_dir=
            output_dir,
    )

    # ========================================================
    # SAVE EXPERIMENT SUMMARY
    # ========================================================

    experiment_summary = {
        "model":
            "ResUNet",

        "base_channels":
            args.base_channels,

        "lambda_l1":
            args.lambda_l1,

        "lambda_ssim":
            args.lambda_ssim,

        "best_epoch":
            best_epoch,

        "best_val_ssim":
            best_val_ssim,

        "best_val_l1":
            best_val_l1,

        "best_val_loss":
            best_val_loss,

        "evaluation_split":
            test_split_name,

        **summary,
    }

    pd.DataFrame(
        [
            experiment_summary
        ]
    ).to_csv(
        output_dir
        / "experiment_summary.csv",

        index=False,
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "EVALUATION RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"N        = "
        f"{summary['n']:,}"
    )

    print(
        f"MAE      = "
        f"{summary['mae_mean']:.6f} "
        f"± "
        f"{summary['mae_std']:.6f}"
    )

    print(
        f"RMSE     = "
        f"{summary['rmse_mean']:.6f} "
        f"± "
        f"{summary['rmse_std']:.6f}"
    )

    print(
        f"PSNR     = "
        f"{summary['psnr_mean']:.3f} "
        f"± "
        f"{summary['psnr_std']:.3f}"
    )

    print(
        f"SSIM     = "
        f"{summary['ssim_mean']:.4f} "
        f"± "
        f"{summary['ssim_std']:.4f}"
    )

    print(
        f"Pearson  = "
        f"{summary['pearson_mean']:.4f} "
        f"± "
        f"{summary['pearson_std']:.4f}"
    )

    print()

    total_runtime_minutes = (
        time.time()
        - run_start
    ) / 60.0

    print(
        f"Total runtime: "
        f"{total_runtime_minutes:.2f} min"
    )

    print(
        "Best checkpoint:"
    )

    print(
        best_checkpoint
    )

    print()

    print(
        "Outputs saved to:"
    )

    print(
        output_dir
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
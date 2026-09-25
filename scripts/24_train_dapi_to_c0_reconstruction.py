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

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET
# ============================================================

class DAPIToC0Dataset(Dataset):

    def __init__(
        self,
        split_csv,
        split,
        augment=False,
    ):

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

        if len(self.df) == 0:
            raise RuntimeError(
                f"No paired MAF/DAPI rows found "
                f"for split={split}"
            )

        print(
            f"DAPIToC0Dataset | split={split}"
        )

        print(
            f"  samples: {len(self.df):,}"
        )

    def __len__(self):

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

        # ----------------------------------------------------
        # Load paired images
        # ----------------------------------------------------

        dapi = tifffile.imread(
            dapi_path
        )

        maf = tifffile.imread(
            maf_path
        )

        # ----------------------------------------------------
        # DAPI must be H x W
        # ----------------------------------------------------

        if dapi.ndim == 3:

            if dapi.shape[-1] == 1:
                dapi = dapi[..., 0]

            else:
                raise ValueError(
                    f"Unexpected DAPI shape: "
                    f"{dapi.shape}"
                )

        # ----------------------------------------------------
        # Original MAF:
        # [C0, C1, C2]
        #
        # Target = C0
        # ----------------------------------------------------

        if (
            maf.ndim != 3
            or maf.shape[-1] < 1
        ):

            raise ValueError(
                f"Unexpected MAF shape: "
                f"{maf.shape}"
            )

        c0 = maf[
            ...,
            0
        ]

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        if dapi.shape != c0.shape:

            raise ValueError(
                f"Shape mismatch:\n"
                f"DAPI: {dapi.shape}\n"
                f"C0:   {c0.shape}\n"
                f"DAPI path: {dapi_path}\n"
                f"MAF path:  {maf_path}"
            )

        # ----------------------------------------------------
        # Convert uint8 -> [0, 1]
        # ----------------------------------------------------

        dapi = dapi.astype(
            np.float32
        ) / 255.0

        c0 = c0.astype(
            np.float32
        ) / 255.0

        # ----------------------------------------------------
        # Paired spatial augmentation
        # ----------------------------------------------------

        if (
            self.augment
            and self.split == "train"
        ):

            # Horizontal flip
            if np.random.rand() < 0.5:

                dapi = np.flip(
                    dapi,
                    axis=1,
                ).copy()

                c0 = np.flip(
                    c0,
                    axis=1,
                ).copy()

            # Vertical flip
            if np.random.rand() < 0.5:

                dapi = np.flip(
                    dapi,
                    axis=0,
                ).copy()

                c0 = np.flip(
                    c0,
                    axis=0,
                ).copy()

            # 0 / 90 / 180 / 270
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

        # ----------------------------------------------------
        # H x W -> 1 x H x W
        # ----------------------------------------------------

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
                row[
                    "filename"
                ],
        }


# ============================================================
# MODEL
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
        base_channels=32,
    ):

        super().__init__()

        b = base_channels

        self.enc1 = DoubleConv(
            1,
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

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            b * 8,
            b * 16,
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
            1,
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
            self.pool(
                e1
            )
        )

        e3 = self.enc3(
            self.pool(
                e2
            )
        )

        e4 = self.enc4(
            self.pool(
                e3
            )
        )

        b = self.bottleneck(
            self.pool(
                e4
            )
        )

        d4 = self.up4(
            b
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

        return torch.sigmoid(
            self.out(
                d1
            )
        )


# ============================================================
# SSIM
# ============================================================

def ssim_torch(
    x,
    y,
    window_size=11,
):

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    padding = (
        window_size // 2
    )

    mu_x = F.avg_pool2d(
        x,
        window_size,
        stride=1,
        padding=padding,
    )

    mu_y = F.avg_pool2d(
        y,
        window_size,
        stride=1,
        padding=padding,
    )

    sigma_x = (
        F.avg_pool2d(
            x * x,
            window_size,
            stride=1,
            padding=padding,
        )
        - mu_x * mu_x
    )

    sigma_y = (
        F.avg_pool2d(
            y * y,
            window_size,
            stride=1,
            padding=padding,
        )
        - mu_y * mu_y
    )

    sigma_xy = (
        F.avg_pool2d(
            x * y,
            window_size,
            stride=1,
            padding=padding,
        )
        - mu_x * mu_y
    )

    numerator = (
        (
            2 * mu_x * mu_y
            + c1
        )
        *
        (
            2 * sigma_xy
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

    return (
        numerator
        / (
            denominator
            + 1e-8
        )
    ).mean()


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

    loss = (
        lambda_l1
        * l1
        +
        lambda_ssim
        * (
            1.0
            - ssim
        )
    )

    return (
        loss,
        l1,
        ssim,
    )


# ============================================================
# TRAIN / VALIDATE
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
        optimizer is not None
    )

    if training:
        model.train()

    else:
        model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_n = 0

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

        batch_size = dapi.shape[0]

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
                prediction,
                c0,
                lambda_l1,
                lambda_ssim,
            )

            if training:

                loss.backward()

                optimizer.step()

        total_loss += (
            float(
                loss.detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_l1 += (
            float(
                l1.detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_ssim += (
            float(
                ssim.detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_n += batch_size

    return {
        "loss":
            total_loss
            / total_n,

        "l1":
            total_l1
            / total_n,

        "ssim":
            total_ssim
            / total_n,
    }


# ============================================================
# TEST METRICS
# ============================================================

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

        for i in range(
            prediction.shape[0]
        ):

            pred = (
                prediction[
                    i,
                    0
                ]
                .cpu()
                .numpy()
            )

            target = (
                c0[
                    i,
                    0
                ]
                .cpu()
                .numpy()
            )

            diff = pred - target

            mae = float(
                np.mean(
                    np.abs(
                        diff
                    )
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        diff ** 2
                    )
                )
            )

            mse = float(
                np.mean(
                    diff ** 2
                )
            )

            if mse <= 1e-12:

                psnr = 120.0

            else:

                psnr = float(
                    10.0
                    * math.log10(
                        1.0 / mse
                    )
                )

            pred_t = torch.from_numpy(
                pred
            )[
                None,
                None
            ].float()

            target_t = torch.from_numpy(
                target
            )[
                None,
                None
            ].float()

            ssim = float(
                ssim_torch(
                    pred_t,
                    target_t,
                )
            )

            p = pred.reshape(
                -1
            )

            t = target.reshape(
                -1
            )

            p0 = p - p.mean()
            t0 = t - t.mean()

            denom = math.sqrt(
                float(
                    np.sum(
                        p0 ** 2
                    )
                )
                *
                float(
                    np.sum(
                        t0 ** 2
                    )
                )
            )

            if denom <= 1e-12:

                pearson = np.nan

            else:

                pearson = float(
                    np.sum(
                        p0 * t0
                    )
                    / denom
                )

            rows.append(
                {
                    "filename":
                        batch[
                            "filename"
                        ][i],

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

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        output_dir
        / "test_metrics_per_image.csv",
        index=False,
    )

    summary = {}

    for metric in [
        "mae",
        "rmse",
        "psnr",
        "ssim",
        "pearson",
    ]:

        values = pd.to_numeric(
            df[
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

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        output_dir
        / "test_metrics_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
    )

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
        default="unet_dapi_to_c0_v1",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve split CSV
    # --------------------------------------------------------

    split_csv = args.split_csv

    if not split_csv.is_absolute():

        split_csv = (
            ROOT
            / split_csv
        )

    if not split_csv.is_file():

        raise FileNotFoundError(
            f"Split CSV not found:\n"
            f"{split_csv}"
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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
    print("=" * 80)
    print("DAPI -> C0 CROSS-CHANNEL RECONSTRUCTION")
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    print(
        f"Split CSV: {split_csv}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = DAPIToC0Dataset(
        split_csv,
        split="train",
        augment=True,
    )

    val_dataset = DAPIToC0Dataset(
        split_csv,
        split="val",
        augment=False,
    )

    try:
        test_dataset = DAPIToC0Dataset(
            split_csv,
            split="test",
            augment=False,
        )

    except RuntimeError:

        print()
        print(
            "WARNING: no test split found."
        )

        print(
            "Using validation split as test set "
            "for this run."
        )

        test_dataset = DAPIToC0Dataset(
            split_csv,
            split="val",
            augment=False,
        )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    loader_kwargs = {
        "batch_size":
            args.batch_size,

        "num_workers":
            args.num_workers,

        "pin_memory":
            torch.cuda.is_available(),
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        base_channels=
            args.base_channels,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=
            args.lr,
        weight_decay=
            args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
    )

    # ============================================================
    # TRAINING
    # ============================================================

    best_val_ssim = float(
        "-inf"
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = -1

    history = []

    best_checkpoint = (
        checkpoint_dir
        / "best_model.pt"
    )

    run_start = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = time.time()

        train_result = run_epoch(
            model,
            train_loader,
            device,
            args.lambda_l1,
            args.lambda_ssim,
            optimizer=
                optimizer,
        )

        val_result = run_epoch(
            model,
            val_loader,
            device,
            args.lambda_l1,
            args.lambda_ssim,
            optimizer=None,
        )

        scheduler.step(
            val_result[
                "loss"
            ]
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        # ========================================================
        # BEST MODEL CRITERION:
        # HIGHEST VALIDATION SSIM
        # ========================================================

        improved = (
            val_result[
                "ssim"
            ]
            > best_val_ssim
        )

        if improved:

            best_val_ssim = (
                val_result[
                    "ssim"
                ]
            )

            best_val_loss = (
                val_result[
                    "loss"
                ]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "val_ssim":
                        best_val_ssim,

                    "val_loss":
                        best_val_loss,

                    "val_l1":
                        val_result[
                            "l1"
                        ],

                    "args":
                        vars(
                            args
                        ),
                },
                best_checkpoint,
            )

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

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_result['loss']:.5f} "
            f"L1={train_result['l1']:.5f} "
            f"SSIM={train_result['ssim']:.4f} | "
            f"val loss={val_result['loss']:.5f} "
            f"L1={val_result['l1']:.5f} "
            f"SSIM={val_result['ssim']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{time.time() - epoch_start:.1f}s"
            + (
                "  <-- BEST SSIM"
                if improved
                else ""
            ),
            flush=True,
        )

    # ============================================================
    # LOAD BEST SSIM MODEL
    # ============================================================

    checkpoint = torch.load(
        best_checkpoint,
        map_location=
            device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best val SSIM: "
        f"{best_val_ssim:.6f}"
    )

    print(
        f"Val loss at best SSIM epoch: "
        f"{best_val_loss:.6f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    summary = evaluate_test(
        model,
        test_loader,
        device,
        output_dir,
    )

    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print(
        f"MAE      = "
        f"{summary['mae_mean']:.6f} "
        f"± {summary['mae_std']:.6f}"
    )

    print(
        f"RMSE     = "
        f"{summary['rmse_mean']:.6f} "
        f"± {summary['rmse_std']:.6f}"
    )

    print(
        f"PSNR     = "
        f"{summary['psnr_mean']:.3f} "
        f"± {summary['psnr_std']:.3f}"
    )

    print(
        f"SSIM     = "
        f"{summary['ssim_mean']:.4f} "
        f"± {summary['ssim_std']:.4f}"
    )

    print(
        f"Pearson  = "
        f"{summary['pearson_mean']:.4f} "
        f"± {summary['pearson_std']:.4f}"
    )

    print()

    print(
        f"Total runtime: "
        f"{(time.time() - run_start) / 60.0:.2f} min"
    )

    print(
        f"Saved to:\n{output_dir}"
    )


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tifffile


# =========================================================
# Constants
# =========================================================

# Original released breast MAF TIFF:
#   C0 = informative
#   C1 = identically zero
#   C2 = informative / strongly DAPI-correlated
#
# We use only the informative channels.
MAF_CHANNEL_INDICES = [0, 2]

MAF_CHANNELS = 2
DAPI_CHANNELS = 1
HE_CHANNELS = 3


# =========================================================
# Basic preprocessing
# =========================================================

def image_to_tensor(
    image,
    expected_channels,
):
    """
    Convert uint8 image to torch tensor in [-1, 1].

    H x W
        ->
    1 x H x W

    H x W x C
        ->
    C x H x W
    """

    if image.ndim == 2:
        image = image[..., None]

    if image.ndim != 3:
        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )

    if image.shape[2] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, "
            f"but got {image.shape}"
        )

    image = image.astype(
        np.float32
    )

    # [0, 255] -> [-1, 1]
    image = (
        image / 127.5
        - 1.0
    )

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1),
    )

    return torch.from_numpy(
        image
    )


def prepare_maf_image(
    image,
):
    """
    Convert original released MAF:

        [C0, C1, C2]

    to:

        [C0, C2]

    because C1 is empty throughout the released
    breast training dataset.
    """

    if image.ndim != 3:
        raise ValueError(
            "MAF image must be H x W x C, "
            f"but got {image.shape}"
        )

    if image.shape[2] < 3:
        raise ValueError(
            "Expected original MAF TIFF to contain "
            f"at least 3 channels, got {image.shape}"
        )

    return image[
        ...,
        MAF_CHANNEL_INDICES
    ]


# =========================================================
# Spatial augmentation
# =========================================================

def paired_spatial_augmentation(
    images,
):
    """
    Apply exactly the same spatial transform to every image
    in the supplied list.

    This is critical for paired MAF/DAPI samples used in
    CrossWave.
    """

    # Horizontal flip
    if np.random.rand() < 0.5:

        images = [
            np.flip(
                image,
                axis=1,
            ).copy()
            for image in images
        ]

    # Vertical flip
    if np.random.rand() < 0.5:

        images = [
            np.flip(
                image,
                axis=0,
            ).copy()
            for image in images
        ]

    # Rotation:
    # 0 / 90 / 180 / 270 degrees
    k = np.random.randint(
        0,
        4,
    )

    if k > 0:

        images = [
            np.rot90(
                image,
                k=k,
                axes=(0, 1),
            ).copy()
            for image in images
        ]

    return images


# =========================================================
# E-CycleGAN training dataset
# =========================================================

class ECycleGANDataset(Dataset):
    """
    Unpaired E-CycleGAN training dataset.

    mode="maf":
        X = 2-channel MAF [C0, C2]
        Y = random H&E

    mode="dapi":
        X = 1-channel recovered DAPI-filter AF
        Y = random H&E

    X and Y are deliberately unpaired.
    """

    def __init__(
        self,
        split_csv,
        split="train",
        mode="maf",
        augment=True,
    ):
        super().__init__()

        if mode not in {
            "maf",
            "dapi",
        }:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        self.mode = mode
        self.split = split
        self.augment = augment

        df = pd.read_csv(
            split_csv
        )

        # -------------------------------------------------
        # X-domain rows
        #
        # domain="maf" rows contain both maf_path and
        # dapi_path because MAF/DAPI are paired.
        # -------------------------------------------------

        self.x_df = (
            df[
                (df["domain"] == "maf")
                &
                (df["split"] == split)
            ]
            .reset_index(
                drop=True
            )
        )

        # -------------------------------------------------
        # Y-domain rows
        # -------------------------------------------------

        self.y_df = (
            df[
                (df["domain"] == "he")
                &
                (df["split"] == split)
            ]
            .reset_index(
                drop=True
            )
        )

        if len(self.x_df) == 0:
            raise RuntimeError(
                f"No X-domain samples found "
                f"for split={split}"
            )

        if len(self.y_df) == 0:
            raise RuntimeError(
                f"No H&E samples found "
                f"for split={split}"
            )

        print(
            f"ECycleGANDataset | "
            f"mode={mode} | "
            f"split={split}"
        )

        print(
            f"  X samples: "
            f"{len(self.x_df):,}"
        )

        print(
            f"  Y samples: "
            f"{len(self.y_df):,}"
        )

        if self.mode == "maf":
            print(
                "  MAF input: [C0, C2] "
                "(2 informative channels)"
            )

    def __len__(
        self,
    ):
        """
        CycleGAN convention:
        use the larger domain to define epoch length.
        """

        return max(
            len(self.x_df),
            len(self.y_df),
        )

    def __getitem__(
        self,
        index,
    ):

        # =================================================
        # X domain
        # =================================================

        x_index = (
            index
            % len(self.x_df)
        )

        x_row = (
            self.x_df.iloc[
                x_index
            ]
        )

        if self.mode == "maf":

            x_path = Path(
                x_row["maf_path"]
            )

            x_image = tifffile.imread(
                x_path
            )

            # Remove empty C1
            x_image = prepare_maf_image(
                x_image
            )

            x_channels = (
                MAF_CHANNELS
            )

        else:

            x_path = Path(
                x_row["dapi_path"]
            )

            x_image = tifffile.imread(
                x_path
            )

            x_channels = (
                DAPI_CHANNELS
            )

        # =================================================
        # Random unpaired H&E
        # =================================================

        y_index = np.random.randint(
            0,
            len(self.y_df),
        )

        y_row = (
            self.y_df.iloc[
                y_index
            ]
        )

        y_path = Path(
            y_row["he_path"]
        )

        y_image = tifffile.imread(
            y_path
        )

        # =================================================
        # Augmentation
        #
        # Because X and Y are unpaired, they can be
        # augmented independently.
        # =================================================

        if (
            self.augment
            and self.split == "train"
        ):

            x_image = (
                paired_spatial_augmentation(
                    [x_image]
                )[0]
            )

            y_image = (
                paired_spatial_augmentation(
                    [y_image]
                )[0]
            )

        # =================================================
        # Tensor conversion
        # =================================================

        x_tensor = image_to_tensor(
            x_image,
            expected_channels=
                x_channels,
        )

        y_tensor = image_to_tensor(
            y_image,
            expected_channels=
                HE_CHANNELS,
        )

        return {
            "x": x_tensor,
            "y": y_tensor,

            "x_filename":
                x_row["filename"],

            "y_filename":
                y_row["filename"],
        }


# =========================================================
# Deterministic single-domain validation dataset
# =========================================================

class SingleDomainDataset(Dataset):
    """
    Deterministic validation dataset.

    Every image is evaluated exactly once.

    domain="x":
        mode="maf"
            -> MAF [C0, C2]

        mode="dapi"
            -> single-channel DAPI

    domain="y":
        -> H&E
    """

    def __init__(
        self,
        split_csv,
        split="val",
        domain="x",
        mode="maf",
    ):
        super().__init__()

        if domain not in {
            "x",
            "y",
        }:
            raise ValueError(
                f"Unsupported domain: {domain}"
            )

        if mode not in {
            "maf",
            "dapi",
        }:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        self.domain = domain
        self.mode = mode
        self.split = split

        df = pd.read_csv(
            split_csv
        )

        # -------------------------------------------------
        # X validation set
        # -------------------------------------------------

        if self.domain == "x":

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

        # -------------------------------------------------
        # Y validation set
        # -------------------------------------------------

        else:

            self.df = (
                df[
                    (df["domain"] == "he")
                    &
                    (df["split"] == split)
                ]
                .reset_index(
                    drop=True
                )
            )

        if len(self.df) == 0:
            raise RuntimeError(
                f"No samples found for "
                f"domain={domain}, "
                f"split={split}"
            )

        print(
            f"SingleDomainDataset | "
            f"domain={domain} | "
            f"mode={mode} | "
            f"split={split}"
        )

        print(
            f"  samples: "
            f"{len(self.df):,}"
        )

        if (
            self.domain == "x"
            and self.mode == "maf"
        ):
            print(
                "  MAF input: [C0, C2] "
                "(2 informative channels)"
            )

    def __len__(
        self,
    ):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):

        row = (
            self.df.iloc[
                index
            ]
        )

        # =================================================
        # X domain
        # =================================================

        if self.domain == "x":

            if self.mode == "maf":

                path = Path(
                    row["maf_path"]
                )

                image = tifffile.imread(
                    path
                )

                image = prepare_maf_image(
                    image
                )

                channels = (
                    MAF_CHANNELS
                )

            else:

                path = Path(
                    row["dapi_path"]
                )

                image = tifffile.imread(
                    path
                )

                channels = (
                    DAPI_CHANNELS
                )

        # =================================================
        # Y domain
        # =================================================

        else:

            path = Path(
                row["he_path"]
            )

            image = tifffile.imread(
                path
            )

            channels = (
                HE_CHANNELS
            )

        tensor = image_to_tensor(
            image,
            expected_channels=
                channels,
        )

        return {
            "image": tensor,
            "filename":
                row["filename"],
        }


# =========================================================
# CrossWave dataset
# =========================================================

class CrossWaveDataset(Dataset):
    """
    CrossWave training dataset.

    Teacher:
        MAF [C0, C2]
        2 channels

    Student:
        recovered DAPI-filter AF
        1 channel

    H&E:
        independently sampled unpaired target-domain image

    MAF and DAPI correspond to the same physical tissue
    patch and therefore always receive identical spatial
    augmentation.
    """

    def __init__(
        self,
        split_csv,
        split="train",
        augment=True,
    ):
        super().__init__()

        self.split = split
        self.augment = augment

        df = pd.read_csv(
            split_csv
        )

        # -------------------------------------------------
        # Paired teacher/student domain
        # -------------------------------------------------

        self.x_df = (
            df[
                (df["domain"] == "maf")
                &
                (df["split"] == split)
            ]
            .reset_index(
                drop=True
            )
        )

        # -------------------------------------------------
        # H&E target domain
        # -------------------------------------------------

        self.y_df = (
            df[
                (df["domain"] == "he")
                &
                (df["split"] == split)
            ]
            .reset_index(
                drop=True
            )
        )

        if len(self.x_df) == 0:
            raise RuntimeError(
                f"No paired MAF/DAPI samples "
                f"for split={split}"
            )

        if len(self.y_df) == 0:
            raise RuntimeError(
                f"No H&E samples "
                f"for split={split}"
            )

        print(
            f"CrossWaveDataset | "
            f"split={split}"
        )

        print(
            f"  paired MAF/DAPI: "
            f"{len(self.x_df):,}"
        )

        print(
            f"  H&E: "
            f"{len(self.y_df):,}"
        )

        print(
            "  Teacher MAF: "
            "[C0, C2] -> 2 channels"
        )

        print(
            "  Student DAPI: "
            "1 channel"
        )

    def __len__(
        self,
    ):

        return max(
            len(self.x_df),
            len(self.y_df),
        )

    def __getitem__(
        self,
        index,
    ):

        # =================================================
        # Paired MAF / DAPI
        # =================================================

        x_index = (
            index
            % len(self.x_df)
        )

        x_row = (
            self.x_df.iloc[
                x_index
            ]
        )

        # -------------------------------------------------
        # Teacher MAF
        # -------------------------------------------------

        maf_path = Path(
            x_row["maf_path"]
        )

        maf = tifffile.imread(
            maf_path
        )

        # Original:
        # [C0, C1, C2]
        #
        # Teacher:
        # [C0, C2]
        maf = prepare_maf_image(
            maf
        )

        # -------------------------------------------------
        # Student DAPI
        # -------------------------------------------------

        dapi_path = Path(
            x_row["dapi_path"]
        )

        dapi = tifffile.imread(
            dapi_path
        )

        # =================================================
        # Random H&E
        # =================================================

        y_index = np.random.randint(
            0,
            len(self.y_df),
        )

        y_row = (
            self.y_df.iloc[
                y_index
            ]
        )

        he_path = Path(
            y_row["he_path"]
        )

        he = tifffile.imread(
            he_path
        )

        # =================================================
        # Augmentation
        # =================================================

        if (
            self.augment
            and self.split == "train"
        ):

            # MAF and DAPI MUST remain aligned
            maf, dapi = (
                paired_spatial_augmentation(
                    [
                        maf,
                        dapi,
                    ]
                )
            )

            # H&E remains independently augmented
            he = (
                paired_spatial_augmentation(
                    [he]
                )[0]
            )

        # =================================================
        # Tensor conversion
        # =================================================

        maf_tensor = image_to_tensor(
            maf,
            expected_channels=
                MAF_CHANNELS,
        )

        dapi_tensor = image_to_tensor(
            dapi,
            expected_channels=
                DAPI_CHANNELS,
        )

        he_tensor = image_to_tensor(
            he,
            expected_channels=
                HE_CHANNELS,
        )

        return {
            "maf": maf_tensor,
            "dapi": dapi_tensor,
            "he": he_tensor,

            "maf_filename":
                x_row["filename"],

            "dapi_filename":
                dapi_path.name,

            "he_filename":
                y_row["filename"],
        }
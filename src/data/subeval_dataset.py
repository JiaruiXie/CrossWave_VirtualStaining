from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


MAF_CHANNEL_INDICES = [0, 2]


def _tile_positions(
    length,
    tile_size,
    stride,
):
    """
    Generate tile starting positions while ensuring
    the final image edge is covered.

    Example:
        length = 1000
        tile   = 256
        stride = 256

    Positions:
        0, 256, 512, 744

    The final tile is anchored to the image edge.
    """

    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)

    if length < tile_size:
        return [0]

    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    final_position = (
        length - tile_size
    )

    if positions[-1] != final_position:
        positions.append(
            final_position
        )

    return positions


class SubEvalTileDataset(Dataset):
    """
    Tile-based dataset for the large TIFF ROIs in:

        data/raw/Sub_eval/AF
        data/raw/Sub_eval/GroudTruth

    domain="af"
        Returns 2-channel MAF [C0, C2]

    domain="he"
        Returns 3-channel real H&E

    All images are normalized from uint8 [0,255]
    to float32 [-1,1].
    """

    def __init__(
        self,
        root,
        domain,
        tile_size=256,
        stride=256,
    ):

        self.root = Path(
            root
        )

        self.domain = str(
            domain
        ).lower()

        self.tile_size = int(
            tile_size
        )

        self.stride = int(
            stride
        )

        if self.domain == "af":

            self.image_dir = (
                self.root
                / "AF"
            )

        elif self.domain in {
            "he",
            "groundtruth",
            "gt",
        }:

            # Preserve the spelling used by the dataset.
            self.image_dir = (
                self.root
                / "GroudTruth"
            )

        else:

            raise ValueError(
                "domain must be 'af' or 'he'. "
                f"Received: {domain}"
            )

        if not self.image_dir.exists():

            raise FileNotFoundError(
                self.image_dir
            )

        self.image_paths = sorted(
            list(
                self.image_dir.glob(
                    "*.tif"
                )
            )
            +
            list(
                self.image_dir.glob(
                    "*.tiff"
                )
            )
        )

        if len(
            self.image_paths
        ) == 0:

            raise RuntimeError(
                f"No TIFF files found in "
                f"{self.image_dir}"
            )

        # Each entry:
        #
        # (
        #   image_path,
        #   x,
        #   y,
        #   valid_width,
        #   valid_height,
        # )
        #
        # valid_width/height are only relevant when
        # an ROI is smaller than 256 pixels.
        self.tiles = []

        self._build_tile_index()

        print(
            "SubEvalTileDataset"
            f" | domain={self.domain}"
        )

        print(
            f"  ROI images: "
            f"{len(self.image_paths):,}"
        )

        print(
            f"  Tiles: "
            f"{len(self.tiles):,}"
        )

        print(
            f"  Tile size: "
            f"{self.tile_size}"
        )

        print(
            f"  Stride: "
            f"{self.stride}"
        )

        if self.domain == "af":

            print(
                "  MAF channels: "
                "[C0, C2]"
            )

    def _build_tile_index(
        self,
    ):

        for image_path in self.image_paths:

            with Image.open(
                image_path
            ) as image:

                width, height = (
                    image.size
                )

            x_positions = (
                _tile_positions(
                    width,
                    self.tile_size,
                    self.stride,
                )
            )

            y_positions = (
                _tile_positions(
                    height,
                    self.tile_size,
                    self.stride,
                )
            )

            for y in y_positions:

                for x in x_positions:

                    valid_width = min(
                        self.tile_size,
                        width - x,
                    )

                    valid_height = min(
                        self.tile_size,
                        height - y,
                    )

                    self.tiles.append(
                        (
                            image_path,
                            x,
                            y,
                            valid_width,
                            valid_height,
                        )
                    )

    def __len__(
        self,
    ):

        return len(
            self.tiles
        )

    def _read_tile(
        self,
        image_path,
        x,
        y,
    ):

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            width, height = (
                image.size
            )

            x2 = min(
                x + self.tile_size,
                width,
            )

            y2 = min(
                y + self.tile_size,
                height,
            )

            crop = image.crop(
                (
                    x,
                    y,
                    x2,
                    y2,
                )
            )

            crop_array = np.array(
                crop,
                dtype=np.uint8,
            )

        # ---------------------------------------------
        # Normally all ROIs are much larger than 256.
        #
        # This handles any unusually small ROI without
        # failing. We pad using the image's edge values.
        # ---------------------------------------------

        h, w = crop_array.shape[
            :2
        ]

        if (
            h < self.tile_size
            or w < self.tile_size
        ):

            pad_h = (
                self.tile_size
                - h
            )

            pad_w = (
                self.tile_size
                - w
            )

            crop_array = np.pad(
                crop_array,
                (
                    (0, pad_h),
                    (0, pad_w),
                    (0, 0),
                ),
                mode="edge",
            )

        return crop_array

    @staticmethod
    def _normalize(
        array,
    ):
        """
        uint8 [0,255]
        ->
        float32 [-1,1]
        """

        array = array.astype(
            np.float32
        )

        array = (
            array / 127.5
        ) - 1.0

        return array

    def __getitem__(
        self,
        index,
    ):

        (
            image_path,
            x,
            y,
            valid_width,
            valid_height,
        ) = self.tiles[
            index
        ]

        image = self._read_tile(
            image_path,
            x,
            y,
        )

        if self.domain == "af":

            # -----------------------------------------
            # Public dataset:
            #
            # informative released channels are C0/C2.
            # C1 is zero in the training MAF dataset.
            # -----------------------------------------

            image = image[
                :,
                :,
                MAF_CHANNEL_INDICES,
            ]

        image = self._normalize(
            image
        )

        # HWC -> CHW
        image = np.transpose(
            image,
            (
                2,
                0,
                1,
            ),
        )

        image = np.ascontiguousarray(
            image
        )

        tensor = torch.from_numpy(
            image
        )

        return {
            "image":
                tensor,

            "path":
                str(
                    image_path
                ),

            "x":
                x,

            "y":
                y,

            "valid_width":
                valid_width,

            "valid_height":
                valid_height,
        }
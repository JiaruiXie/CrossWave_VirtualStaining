"""
Reconstruct full virtual H&E images for the Sub_eval AF dataset using
the frozen multispectral teacher.

Default teacher:
    outputs/maf_teacher_inverted_y_adaptive_scratch/checkpoints/best_fid.pt

Input:
    data/raw/Sub_eval/AF

Output:
    outputs/maf_teacher_inverted_y_adaptive_scratch/sub_eval_vhe

The released MAF TIFFs are stored as 3-channel images, but the trained
teacher uses the two informative channels [C0, C2].

The teacher was trained in an inverted H&E model domain:
    y_model = -y_physical

Therefore inference restores physical H&E with:
    y_physical = -G_XY(x)

Large Sub_eval ROIs are reconstructed tile-by-tile and stitched back to
their original image size. By default, 256x256 tiles with stride 224 are
used to provide overlap and reduce tile-boundary artifacts.

This script saves images only. It does not generate plots.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

import torch


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.models.ecyclegan import ECycleGAN


Image.MAX_IMAGE_PIXELS = None


DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "maf_teacher_stable_b_inverted_y_v2"
    / "checkpoints"
    / "best_fid.pt"
)

DEFAULT_INPUT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Sub_eval"
    / "AF"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "maf_teacher_stable_b_inverted_y_v2"
    / "sub_eval_vhe"
)


# =====================================================================
# Utilities
# =====================================================================


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def find_images(folder: Path) -> list[Path]:
    valid_suffixes = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    }

    files = [
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in valid_suffixes
        )
    ]

    return sorted(files)


def tile_positions(
    length: int,
    tile_size: int,
    stride: int,
) -> list[int]:
    """
    Return tile start positions while ensuring that the final tile reaches
    the far image edge.
    """

    if length <= tile_size:
        return [0]

    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    last_position = (
        length - tile_size
    )

    if positions[-1] != last_position:
        positions.append(
            last_position
        )

    return positions


def make_blend_window(
    tile_size: int,
) -> np.ndarray:
    """
    Soft spatial weighting for overlap stitching.

    A small nonzero floor is retained so pixels on the outer image boundary
    still receive positive weight.
    """

    if tile_size <= 2:
        return np.ones(
            (
                tile_size,
                tile_size,
            ),
            dtype=np.float32,
        )

    one_d = np.hanning(
        tile_size
    ).astype(
        np.float32
    )

    two_d = np.outer(
        one_d,
        one_d,
    )

    if float(
        two_d.max()
    ) > 0:
        two_d /= (
            two_d.max()
        )

    window = (
        0.10
        + 0.90
        * two_d
    )

    return window.astype(
        np.float32
    )


def load_maf_image(
    path: Path,
) -> np.ndarray:
    """
    Load released Sub_eval MAF as uint8 H x W x 3.
    """

    with Image.open(
        path
    ) as image:
        image = image.convert(
            "RGB"
        )

        array = np.asarray(
            image,
            dtype=np.uint8,
        )

    if (
        array.ndim != 3
        or array.shape[2] != 3
    ):
        raise RuntimeError(
            f"Expected HxWx3 image, got {array.shape} for {path}"
        )

    return array


def extract_teacher_input(
    rgb_tile: np.ndarray,
) -> torch.Tensor:
    """
    Released processed MAF storage:
        channel 0 -> C0
        channel 1 -> non-informative / zero
        channel 2 -> C2 (DAPI-filter AF)

    Teacher input:
        [C0, C2]

    uint8 [0,255] -> float32 [-1,1]
    """

    informative = rgb_tile[
        :,
        :,
        [
            0,
            2,
        ],
    ].astype(
        np.float32
    )

    informative = (
        informative
        / 127.5
        - 1.0
    )

    informative = np.transpose(
        informative,
        (
            2,
            0,
            1,
        ),
    )

    return torch.from_numpy(
        informative
    )


def pad_tile(
    tile: np.ndarray,
    tile_size: int,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    """
    Edge-pad small boundary/ROI tiles to tile_size x tile_size.
    """

    height, width = tile.shape[
        :2
    ]

    if (
        height == tile_size
        and width == tile_size
    ):
        return (
            tile,
            height,
            width,
        )

    pad_bottom = (
        tile_size
        - height
    )

    pad_right = (
        tile_size
        - width
    )

    padded = np.pad(
        tile,
        (
            (
                0,
                pad_bottom,
            ),
            (
                0,
                pad_right,
            ),
            (
                0,
                0,
            ),
        ),
        mode="edge",
    )

    return (
        padded,
        height,
        width,
    )


# =====================================================================
# Model loading
# =====================================================================


def load_teacher(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    ECycleGAN,
    dict,
    int,
    float,
    bool,
]:
    print()
    print(
        "=" * 72
    )

    print(
        "LOADING TEACHER"
    )

    print(
        "=" * 72
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        not isinstance(
            checkpoint,
            dict,
        )
        or "model_state_dict"
        not in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    config = checkpoint.get(
        "config",
        {},
    )

    model_cfg = config.get(
        "model",
        {},
    )

    data_cfg = config.get(
        "data",
        {},
    )

    x_channels = int(
        model_cfg.get(
            "x_channels",
            2,
        )
    )

    y_channels = int(
        model_cfg.get(
            "y_channels",
            3,
        )
    )

    base_channels = int(
        model_cfg.get(
            "base_channels",
            64,
        )
    )

    num_res_blocks = int(
        model_cfg.get(
            "num_res_blocks",
            9,
        )
    )

    padding_type = model_cfg.get(
        "padding_type",
        "reflect",
    )

    if x_channels != 2:
        raise RuntimeError(
            "Expected the multispectral teacher to have x_channels=2, "
            f"but checkpoint reports {x_channels}."
        )

    model = ECycleGAN(
        x_channels=x_channels,
        y_channels=y_channels,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
        padding_type=padding_type,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )

    best_fid = float(
        checkpoint.get(
            "best_fid",
            np.nan,
        )
    )

    invert_y_domain = bool(
        data_cfg.get(
            "invert_y_domain",
            True,
        )
    )

    print(
        "Checkpoint epoch:",
        epoch,
    )

    print(
        "Best FID:",
        best_fid,
    )

    print(
        "Teacher X channels:",
        x_channels,
    )

    print(
        "Invert Y domain:",
        invert_y_domain,
    )

    print(
        "Device:",
        device,
    )

    print(
        "=" * 72
    )

    print()

    return (
        model,
        config,
        epoch,
        best_fid,
        invert_y_domain,
    )


# =====================================================================
# Inference
# =====================================================================


@torch.inference_mode()
def infer_batch(
    model: ECycleGAN,
    batch_tensor: torch.Tensor,
    device: torch.device,
    invert_y_domain: bool,
    use_amp: bool,
) -> np.ndarray:
    """
    Returns B x H x W x 3 uint8 physical H&E.
    """

    batch_tensor = batch_tensor.to(
        device,
        non_blocking=True,
    )

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        fake_y_model = model.G_XY(
            batch_tensor
        )

    # Teacher was trained with:
    #     y_model = -y_physical
    #
    # Restore physical H&E before saving.
    if invert_y_domain:
        fake_y = (
            -fake_y_model
        )
    else:
        fake_y = (
            fake_y_model
        )

    fake_y = torch.clamp(
        fake_y,
        -1.0,
        1.0,
    )

    fake_y = (
        fake_y
        + 1.0
    ) / 2.0

    fake_y = (
        fake_y
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    fake_y = np.transpose(
        fake_y,
        (
            0,
            2,
            3,
            1,
        ),
    )

    fake_y = np.clip(
        np.rint(
            fake_y
            * 255.0
        ),
        0,
        255,
    ).astype(
        np.uint8
    )

    return fake_y


@torch.inference_mode()
def reconstruct_roi(
    maf_image: np.ndarray,
    model: ECycleGAN,
    device: torch.device,
    tile_size: int,
    stride: int,
    batch_size: int,
    invert_y_domain: bool,
    use_amp: bool,
) -> tuple[
    np.ndarray,
    int,
]:
    """
    Tile a complete Sub_eval AF ROI, perform teacher inference, and
    overlap-stitch the virtual H&E result back to the original size.
    """

    image_height, image_width = maf_image.shape[
        :2
    ]

    y_positions = tile_positions(
        image_height,
        tile_size,
        stride,
    )

    x_positions = tile_positions(
        image_width,
        tile_size,
        stride,
    )

    coordinates = [
        (
            y,
            x,
        )
        for y in y_positions
        for x in x_positions
    ]

    blend_window = make_blend_window(
        tile_size
    )

    accumulator = np.zeros(
        (
            image_height,
            image_width,
            3,
        ),
        dtype=np.float32,
    )

    weights = np.zeros(
        (
            image_height,
            image_width,
            1,
        ),
        dtype=np.float32,
    )

    for batch_start in range(
        0,
        len(
            coordinates
        ),
        batch_size,
    ):
        batch_coordinates = coordinates[
            batch_start:
            batch_start
            + batch_size
        ]

        tensors = []
        valid_shapes = []

        for (
            y,
            x,
        ) in batch_coordinates:
            tile = maf_image[
                y:
                min(
                    y + tile_size,
                    image_height,
                ),
                x:
                min(
                    x + tile_size,
                    image_width,
                ),
                :,
            ]

            (
                tile,
                valid_height,
                valid_width,
            ) = pad_tile(
                tile,
                tile_size,
            )

            tensor = extract_teacher_input(
                tile
            )

            tensors.append(
                tensor
            )

            valid_shapes.append(
                (
                    valid_height,
                    valid_width,
                )
            )

        input_batch = torch.stack(
            tensors,
            dim=0,
        )

        predictions = infer_batch(
            model=model,
            batch_tensor=input_batch,
            device=device,
            invert_y_domain=invert_y_domain,
            use_amp=use_amp,
        )

        for index, (
            y,
            x,
        ) in enumerate(
            batch_coordinates
        ):
            (
                valid_height,
                valid_width,
            ) = valid_shapes[
                index
            ]

            prediction = predictions[
                index,
                :
                    valid_height,
                :
                    valid_width,
                :,
            ].astype(
                np.float32
            )

            window = blend_window[
                :
                    valid_height,
                :
                    valid_width,
            ][
                :,
                :,
                None,
            ]

            accumulator[
                y:
                    y
                    + valid_height,
                x:
                    x
                    + valid_width,
                :,
            ] += (
                prediction
                * window
            )

            weights[
                y:
                    y
                    + valid_height,
                x:
                    x
                    + valid_width,
                :,
            ] += (
                window
            )

    reconstructed = (
        accumulator
        / np.maximum(
            weights,
            1e-8,
        )
    )

    reconstructed = np.clip(
        np.rint(
            reconstructed
        ),
        0,
        255,
    ).astype(
        np.uint8
    )

    return (
        reconstructed,
        len(
            coordinates
        ),
    )


# =====================================================================
# Saving
# =====================================================================


def save_vhe(
    image: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pil_image = Image.fromarray(
        image,
        mode="RGB",
    )

    suffix = output_path.suffix.lower()

    if suffix in {
        ".tif",
        ".tiff",
    }:
        pil_image.save(
            output_path,
            compression="tiff_deflate",
        )
    else:
        pil_image.save(
            output_path,
        )


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Sub_eval virtual H&E images using "
            "the frozen multispectral teacher."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            DEFAULT_CHECKPOINT
        ),
        help=(
            "Path to teacher checkpoint."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(
            DEFAULT_INPUT_DIR
        ),
        help=(
            "Sub_eval AF directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory for reconstructed teacher virtual H&E images."
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help=(
            "Inference tile size. Default: 256."
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=224,
        help=(
            "Inference stride. Default: 224 for overlapping reconstruction."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Teacher inference batch size. Default: 16."
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help=(
            "Disable CUDA mixed-precision inference."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing reconstructed TIFFs."
        ),
    )

    args = parser.parse_args()

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    input_dir = resolve_path(
        args.input_dir
    )

    output_dir = resolve_path(
        args.output_dir
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {checkpoint_path}"
        )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Sub_eval AF directory not found: {input_dir}"
        )

    if args.tile_size <= 0:
        raise ValueError(
            "--tile-size must be > 0."
        )

    if args.stride <= 0:
        raise ValueError(
            "--stride must be > 0."
        )

    if args.stride > args.tile_size:
        raise ValueError(
            "--stride should not be larger than --tile-size, "
            "otherwise gaps would appear in the reconstructed image."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    (
        teacher,
        teacher_config,
        teacher_epoch,
        teacher_best_fid,
        invert_y_domain,
    ) = load_teacher(
        checkpoint_path=
            checkpoint_path,
        device=
            device,
    )

    image_paths = find_images(
        input_dir
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in {input_dir}"
        )

    print(
        "Sub_eval AF images:",
        len(
            image_paths
        ),
    )

    print(
        "Tile size:",
        args.tile_size,
    )

    print(
        "Stride:",
        args.stride,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "AMP:",
        not args.no_amp,
    )

    print(
        "Output:",
        output_dir,
    )

    print()

    manifest_rows = []

    overall_start = time.time()

    for image_index, input_path in enumerate(
        image_paths,
        start=1,
    ):
        output_path = (
            output_dir
            / (
                input_path.stem
                + "_teacher_vHE.tif"
            )
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            print(
                f"[{image_index:02d}/{len(image_paths):02d}] "
                f"SKIP existing: {output_path.name}"
            )

            continue

        print(
            f"[{image_index:02d}/{len(image_paths):02d}] "
            f"{input_path.name}"
        )

        start = time.time()

        maf_image = load_maf_image(
            input_path
        )

        reconstructed, tile_count = reconstruct_roi(
            maf_image=maf_image,
            model=teacher,
            device=device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            invert_y_domain=invert_y_domain,
            use_amp=(
                not args.no_amp
            ),
        )

        save_vhe(
            reconstructed,
            output_path,
        )

        elapsed = (
            time.time()
            - start
        )

        print(
            f"    size: "
            f"{maf_image.shape[1]} x {maf_image.shape[0]}"
        )

        print(
            f"    tiles: {tile_count}"
        )

        print(
            f"    saved: {output_path.name}"
        )

        print(
            f"    time: {elapsed:.1f} s"
        )

        manifest_rows.append(
            {
                "input_file":
                    input_path.name,
                "output_file":
                    output_path.name,
                "width":
                    int(
                        maf_image.shape[1]
                    ),
                "height":
                    int(
                        maf_image.shape[0]
                    ),
                "tile_size":
                    int(
                        args.tile_size
                    ),
                "stride":
                    int(
                        args.stride
                    ),
                "tile_count":
                    int(
                        tile_count
                    ),
                "seconds":
                    float(
                        elapsed
                    ),
                "teacher_checkpoint":
                    str(
                        checkpoint_path
                    ),
                "teacher_epoch":
                    int(
                        teacher_epoch
                    ),
                "teacher_best_fid":
                    float(
                        teacher_best_fid
                    ),
                "invert_y_domain":
                    bool(
                        invert_y_domain
                    ),
            }
        )

    manifest_path = (
        output_dir
        / "reconstruction_manifest.csv"
    )

    if manifest_rows:
        fieldnames = list(
            manifest_rows[0].keys()
        )

        with manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                manifest_rows
            )

    total_elapsed = (
        time.time()
        - overall_start
    )

    print()
    print(
        "=" * 72
    )

    print(
        "RECONSTRUCTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Images reconstructed:",
        len(
            manifest_rows
        ),
    )

    print(
        "Output directory:",
        output_dir,
    )

    if manifest_rows:
        print(
            "Manifest:",
            manifest_path,
        )

    print(
        f"Total time: "
        f"{total_elapsed / 60.0:.1f} min"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()

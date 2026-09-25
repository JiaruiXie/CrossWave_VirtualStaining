from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ecyclegan import ECycleGAN

Image.MAX_IMAGE_PIXELS = None

DEFAULT_EXPERIMENT_DIR = ROOT / "outputs" / "dapi_baseline_v1"
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT_DIR / "checkpoints" / "best_fid.pt"
DEFAULT_INPUT_DIR = ROOT / "data" / "raw" / "Sub_eval" / "AF"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "sub_eval_vhe"


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def find_images(folder):
    suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )


def tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    positions = list(
        range(0, length - tile_size + 1, stride)
    )

    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)

    return positions


def make_blend_window(tile_size):
    if tile_size <= 2:
        return np.ones(
            (tile_size, tile_size),
            dtype=np.float32,
        )

    one_d = np.hanning(tile_size).astype(np.float32)
    window = np.outer(one_d, one_d)

    if float(window.max()) > 0:
        window /= window.max()

    return (0.10 + 0.90 * window).astype(np.float32)


def load_maf_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)

    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(
            f"Expected HxWx3 AF image, got {array.shape}: {path}"
        )

    return array


def extract_baseline_input(rgb_tile, source_channel_index=2):
    single_band = rgb_tile[
        :, :, source_channel_index:source_channel_index + 1
    ].astype(np.float32)

    single_band = single_band / 127.5 - 1.0
    single_band = np.transpose(single_band, (2, 0, 1))

    return torch.from_numpy(single_band)


def pad_tile(tile, tile_size):
    height, width = tile.shape[:2]

    if height == tile_size and width == tile_size:
        return tile, height, width

    padded = np.pad(
        tile,
        (
            (0, tile_size - height),
            (0, tile_size - width),
            (0, 0),
        ),
        mode="edge",
    )

    return padded, height, width


def load_baseline(checkpoint_path, device):
    print("=" * 72)
    print("LOADING DAPI BASELINE")
    print("=" * 72)
    print("Checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    x_channels = int(model_cfg.get("x_channels", 1))
    y_channels = int(model_cfg.get("y_channels", 3))
    base_channels = int(model_cfg.get("base_channels", 64))
    num_res_blocks = int(model_cfg.get("num_res_blocks", 9))
    padding_type = model_cfg.get("padding_type", "reflect")

    if x_channels != 1:
        raise RuntimeError(
            f"Expected x_channels=1, got {x_channels}."
        )

    model = ECycleGAN(
        x_channels=x_channels,
        y_channels=y_channels,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
        padding_type=padding_type,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    epoch = int(checkpoint.get("epoch", -1))
    best_fid = float(checkpoint.get("best_fid", np.nan))
    invert_y_domain = bool(
        data_cfg.get("invert_y_domain", True)
    )

    print("Checkpoint epoch:", epoch)
    print("Best FID:", best_fid)
    print("Input channels:", x_channels)
    print("Input used:", "C2 only")
    print("Invert Y domain:", invert_y_domain)
    print("Device:", device)
    print("=" * 72)

    return model, epoch, best_fid, invert_y_domain


@torch.inference_mode()
def infer_batch(
    model,
    batch_tensor,
    device,
    invert_y_domain,
    use_amp,
):
    batch_tensor = batch_tensor.to(
        device,
        non_blocking=True,
    )

    amp_enabled = use_amp and device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        fake_y_model = model.G_XY(batch_tensor)

    if invert_y_domain:
        fake_y = -fake_y_model
    else:
        fake_y = fake_y_model

    fake_y = torch.clamp(fake_y, -1.0, 1.0)
    fake_y = (fake_y + 1.0) / 2.0

    array = (
        fake_y.detach()
        .float()
        .cpu()
        .numpy()
    )

    array = np.transpose(array, (0, 2, 3, 1))

    return np.clip(
        np.rint(array * 255.0),
        0,
        255,
    ).astype(np.uint8)


@torch.inference_mode()
def reconstruct_roi(
    maf_image,
    model,
    device,
    source_channel_index,
    tile_size,
    stride,
    batch_size,
    invert_y_domain,
    use_amp,
):
    height, width = maf_image.shape[:2]

    y_positions = tile_positions(
        height,
        tile_size,
        stride,
    )
    x_positions = tile_positions(
        width,
        tile_size,
        stride,
    )

    coordinates = [
        (y, x)
        for y in y_positions
        for x in x_positions
    ]

    blend_window = make_blend_window(tile_size)

    accumulator = np.zeros(
        (height, width, 3),
        dtype=np.float32,
    )
    weights = np.zeros(
        (height, width, 1),
        dtype=np.float32,
    )

    for batch_start in range(
        0,
        len(coordinates),
        batch_size,
    ):
        batch_coordinates = coordinates[
            batch_start:batch_start + batch_size
        ]

        tensors = []
        valid_shapes = []

        for y, x in batch_coordinates:
            tile = maf_image[
                y:min(y + tile_size, height),
                x:min(x + tile_size, width),
                :,
            ]

            tile, valid_h, valid_w = pad_tile(
                tile,
                tile_size,
            )

            tensors.append(
                extract_baseline_input(
                    tile,
                    source_channel_index,
                )
            )
            valid_shapes.append((valid_h, valid_w))

        batch_tensor = torch.stack(tensors, dim=0)

        predictions = infer_batch(
            model,
            batch_tensor,
            device,
            invert_y_domain,
            use_amp,
        )

        for index, (y, x) in enumerate(batch_coordinates):
            valid_h, valid_w = valid_shapes[index]

            prediction = predictions[
                index,
                :valid_h,
                :valid_w,
                :,
            ].astype(np.float32)

            window = blend_window[
                :valid_h,
                :valid_w,
            ][:, :, None]

            accumulator[
                y:y + valid_h,
                x:x + valid_w,
                :,
            ] += prediction * window

            weights[
                y:y + valid_h,
                x:x + valid_w,
                :,
            ] += window

    reconstructed = accumulator / np.maximum(
        weights,
        1e-8,
    )

    reconstructed = np.clip(
        np.rint(reconstructed),
        0,
        255,
    ).astype(np.uint8)

    return reconstructed, len(coordinates)


def save_vhe(image, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        image,
        mode="RGB",
    ).save(
        output_path,
        compression="tiff_deflate",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--source-channel-index",
        type=int,
        default=2,
        choices=[0, 1, 2],
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if args.stride > args.tile_size:
        raise ValueError(
            "--stride must be <= --tile-size."
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
        model,
        checkpoint_epoch,
        checkpoint_best_fid,
        invert_y_domain,
    ) = load_baseline(
        checkpoint_path,
        device,
    )

    image_paths = find_images(input_dir)

    if not image_paths:
        raise RuntimeError(
            f"No images found in {input_dir}"
        )

    print("Sub_eval AF images:", len(image_paths))
    print("Source TIFF channel index:", args.source_channel_index)
    print("Tile size:", args.tile_size)
    print("Stride:", args.stride)
    print("Batch size:", args.batch_size)
    print("Output:", output_dir)
    print()

    manifest_rows = []
    overall_start = time.time()

    for image_index, input_path in enumerate(
        image_paths,
        start=1,
    ):
        output_path = (
            output_dir
            / f"{input_path.stem}_dapi_baseline_vHE.tif"
        )

        if output_path.exists() and not args.overwrite:
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

        maf_image = load_maf_image(input_path)

        reconstructed, tile_count = reconstruct_roi(
            maf_image=maf_image,
            model=model,
            device=device,
            source_channel_index=args.source_channel_index,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            invert_y_domain=invert_y_domain,
            use_amp=not args.no_amp,
        )

        save_vhe(
            reconstructed,
            output_path,
        )

        elapsed = time.time() - start

        print(
            f"    size: {maf_image.shape[1]} x "
            f"{maf_image.shape[0]}"
        )
        print(f"    tiles: {tile_count}")
        print(f"    saved: {output_path.name}")
        print(f"    time: {elapsed:.1f} s")

        manifest_rows.append(
            {
                "input_file": input_path.name,
                "output_file": output_path.name,
                "width": int(maf_image.shape[1]),
                "height": int(maf_image.shape[0]),
                "source_channel_index": int(
                    args.source_channel_index
                ),
                "tile_size": int(args.tile_size),
                "stride": int(args.stride),
                "tile_count": int(tile_count),
                "seconds": float(elapsed),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_best_fid": checkpoint_best_fid,
                "invert_y_domain": invert_y_domain,
            }
        )

    manifest_path = (
        output_dir
        / "reconstruction_manifest.csv"
    )

    if manifest_rows:
        with manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    manifest_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    print()
    print("=" * 72)
    print("DAPI BASELINE SUB_EVAL RECONSTRUCTION COMPLETE")
    print("=" * 72)
    print("Images reconstructed:", len(manifest_rows))
    print("Output directory:", output_dir)

    if manifest_rows:
        print("Manifest:", manifest_path)

    print(
        f"Total time: "
        f"{(time.time() - overall_start) / 60.0:.1f} min"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

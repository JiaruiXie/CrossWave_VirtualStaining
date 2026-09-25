"""
Generate full virtual H&E images for Sub_eval using the CrossWave v3 student.

Default experiment:
    outputs/crosswave_dapi_student_lr_v3

Default checkpoint:
    outputs/crosswave_dapi_student_lr_v3/checkpoints/best_fid.pt

Student input is exactly the same single component used during training:
    released Sub_eval MAF TIFF -> C2 (array index 2) -> [-1, 1]

The model was trained in the inverted-Y domain, so the generated model-domain
H&E is multiplied by -1 before conversion back to uint8 physical H&E.

No plots are generated.
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


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ecyclegan import ECycleGAN

Image.MAX_IMAGE_PIXELS = None

EXPERIMENT_DIR = ROOT / "outputs" / "crosswave_dapi_student_newTeacher_v2"
DEFAULT_CHECKPOINT = EXPERIMENT_DIR / "checkpoints" / "best_fid.pt"
DEFAULT_INPUT_DIR = ROOT / "data" / "raw" / "Sub_eval" / "AF"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "sub_eval_vhe"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def find_images(folder: Path) -> list[Path]:
    suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )


def positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    out = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if out[-1] != last:
        out.append(last)
    return out


def blend_window(tile_size: int) -> np.ndarray:
    """Hann-like overlap weight with nonzero border weight."""
    if tile_size <= 2:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    w = np.hanning(tile_size).astype(np.float32)
    w = np.outer(w, w)
    if float(w.max()) > 0:
        w /= float(w.max())
    return (0.10 + 0.90 * w).astype(np.float32)


def load_af(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"Expected HxWx3 image, got {array.shape}: {path}")
    return array


def make_student_tensor(rgb_tile: np.ndarray) -> torch.Tensor:
    """
    CrossWave student uses C2 only.

    The training MAF loader returned [C0, C2], and the student sliced index 1
    from that 2-channel tensor. In the released RGB-like TIFF, C2 is stored at
    array index 2.
    """
    x = rgb_tile[:, :, 2:3].astype(np.float32)
    x = x / 127.5 - 1.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def pad_to_tile(tile: np.ndarray, tile_size: int):
    h, w = tile.shape[:2]
    if h == tile_size and w == tile_size:
        return tile, h, w
    padded = np.pad(
        tile,
        ((0, tile_size - h), (0, tile_size - w), (0, 0)),
        mode="edge",
    )
    return padded, h, w


def load_student(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint is missing 'model_state_dict'.")

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
            f"Expected CrossWave student x_channels=1, got {x_channels}."
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
    for p in model.parameters():
        p.requires_grad = False

    epoch = int(checkpoint.get("epoch", -1))
    best_fid = float(checkpoint.get("best_fid", np.nan))
    invert_y_domain = bool(data_cfg.get("invert_y_domain", True))

    print("=" * 72)
    print("LOADING CROSSWAVE V3 STUDENT")
    print("=" * 72)
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", epoch)
    print("Best FID:", best_fid)
    print("Student X channels:", x_channels)
    print("Student input: released TIFF C2 only (array index 2)")
    print("Invert Y domain:", invert_y_domain)
    print("Device:", device)
    print("=" * 72)
    print()

    return model, epoch, best_fid, invert_y_domain


@torch.inference_mode()
def predict_batch(
    model: ECycleGAN,
    batch: torch.Tensor,
    device: torch.device,
    invert_y_domain: bool,
    use_amp: bool,
) -> np.ndarray:
    batch = batch.to(device, non_blocking=True)
    amp_enabled = use_amp and device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        fake_y_model = model.G_XY(batch)

    fake_y = -fake_y_model if invert_y_domain else fake_y_model
    fake_y = torch.clamp(fake_y, -1.0, 1.0)
    fake_y = (fake_y + 1.0) / 2.0

    array = fake_y.detach().float().cpu().numpy()
    array = np.transpose(array, (0, 2, 3, 1))
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


@torch.inference_mode()
def reconstruct_roi(
    af: np.ndarray,
    model: ECycleGAN,
    device: torch.device,
    tile_size: int,
    stride: int,
    batch_size: int,
    invert_y_domain: bool,
    use_amp: bool,
):
    h, w = af.shape[:2]
    ys = positions(h, tile_size, stride)
    xs = positions(w, tile_size, stride)
    coords = [(y, x) for y in ys for x in xs]

    window = blend_window(tile_size)
    accumulator = np.zeros((h, w, 3), dtype=np.float32)
    weights = np.zeros((h, w, 1), dtype=np.float32)

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        tensors = []
        shapes = []

        for y, x in batch_coords:
            tile = af[
                y:min(y + tile_size, h),
                x:min(x + tile_size, w),
                :,
            ]
            tile, valid_h, valid_w = pad_to_tile(tile, tile_size)
            tensors.append(make_student_tensor(tile))
            shapes.append((valid_h, valid_w))

        predictions = predict_batch(
            model=model,
            batch=torch.stack(tensors, dim=0),
            device=device,
            invert_y_domain=invert_y_domain,
            use_amp=use_amp,
        )

        for i, (y, x) in enumerate(batch_coords):
            valid_h, valid_w = shapes[i]
            pred = predictions[i, :valid_h, :valid_w].astype(np.float32)
            local_w = window[:valid_h, :valid_w, None]

            accumulator[y:y + valid_h, x:x + valid_w] += pred * local_w
            weights[y:y + valid_h, x:x + valid_w] += local_w

    output = accumulator / np.maximum(weights, 1e-8)
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    return output, len(coords)


def save_tiff(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(
        path,
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
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.tile_size <= 0 or args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("tile-size, stride, and batch-size must be positive.")
    if args.stride > args.tile_size:
        raise ValueError("stride must be <= tile-size.")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, epoch, best_fid, invert_y_domain = load_student(
        checkpoint_path,
        device,
    )

    images = find_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    print("Sub_eval AF images:", len(images))
    print("Tile size:", args.tile_size)
    print("Stride:", args.stride)
    print("Batch size:", args.batch_size)
    print("AMP:", not args.no_amp)
    print("Output:", output_dir)
    print()

    rows = []
    total_start = time.time()

    for index, input_path in enumerate(images, start=1):
        output_path = output_dir / f"{input_path.stem}_crosswave_v3_vHE.tif"

        if output_path.exists() and not args.overwrite:
            print(f"[{index:02d}/{len(images):02d}] SKIP {output_path.name}")
            continue

        print(f"[{index:02d}/{len(images):02d}] {input_path.name}")
        start = time.time()

        af = load_af(input_path)
        vhe, tile_count = reconstruct_roi(
            af=af,
            model=model,
            device=device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            invert_y_domain=invert_y_domain,
            use_amp=not args.no_amp,
        )
        save_tiff(vhe, output_path)

        elapsed = time.time() - start
        print(f"    size: {af.shape[1]} x {af.shape[0]}")
        print(f"    tiles: {tile_count}")
        print(f"    saved: {output_path.name}")
        print(f"    time: {elapsed:.1f} s")

        rows.append({
            "input_file": input_path.name,
            "output_file": output_path.name,
            "width": int(af.shape[1]),
            "height": int(af.shape[0]),
            "student_input": "C2_only",
            "released_tiff_channel_index": 2,
            "tile_size": int(args.tile_size),
            "stride": int(args.stride),
            "tile_count": int(tile_count),
            "seconds": float(elapsed),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(epoch),
            "checkpoint_best_fid": float(best_fid),
            "invert_y_domain": bool(invert_y_domain),
        })

    if rows:
        manifest_path = output_dir / "reconstruction_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Manifest:", manifest_path)

    print()
    print("=" * 72)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 72)
    print("Images reconstructed:", len(rows))
    print("Output directory:", output_dir)
    print(f"Total time: {(time.time() - total_start) / 60.0:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()

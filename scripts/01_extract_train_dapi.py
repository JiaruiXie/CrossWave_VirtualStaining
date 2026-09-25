from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile
import openslide


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(
    0,
    str(ROOT)
)


from src.data.filename_parser import (
    parse_training_maf_filename,
    BREAST_SAMPLE_MAP,
)

from src.data.dapi_extraction import (
    extract_dapi_patch,
)


MAF_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Training"
    / "trainA"
)

RAW_AF_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Initial_data"
    / "AF"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "train"
    / "dapi"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "train_dapi_manifest.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def normalized_cross_correlation(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a_std = a.std()
    b_std = b.std()

    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0

    a = (a - a.mean()) / a_std
    b = (b - b.mean()) / b_std

    return float(
        np.mean(a * b)
    )


def main():

    maf_files = sorted(
        MAF_DIR.glob("*.tif")
    )

    print(
        f"Found {len(maf_files):,} MAF patches"
    )

    slides = {}
    records = []

    try:

        for i, maf_path in enumerate(
            maf_files,
            start=1,
        ):

            info = parse_training_maf_filename(
                maf_path
            )

            if info is None:

                records.append({
                    "maf_filename": maf_path.name,
                    "valid": False,
                    "reason": "filename_parse_failed",
                })

                continue

            train_sample = info["sample"]

            raw_sample = BREAST_SAMPLE_MAP.get(
                train_sample
            )

            if raw_sample is None:

                records.append({
                    "maf_filename": maf_path.name,
                    "train_sample": train_sample,
                    "valid": False,
                    "reason": "sample_mapping_missing",
                })

                continue

            # Cache slides
            if raw_sample not in slides:

                slide_path = (
                    RAW_AF_DIR
                    / f"sample_{raw_sample:02d}.mrxs"
                )

                slides[raw_sample] = (
                    openslide.OpenSlide(
                        str(slide_path)
                    )
                )

            slide = slides[raw_sample]

            dapi, raw_x, raw_y = (
                extract_dapi_patch(
                    slide=slide,
                    relative_x=info["global_x"],
                    relative_y=info["global_y"],
                    width=256,
                    height=256,
                )
            )

            # Load released teacher MAF
            maf = tifffile.imread(
                maf_path
            )

            if (
                maf.ndim != 3
                or maf.shape[0] != 256
                or maf.shape[1] != 256
            ):
                records.append({
                    "maf_filename": maf_path.name,
                    "train_sample": train_sample,
                    "raw_sample": raw_sample,
                    "valid": False,
                    "reason": (
                        f"unexpected_maf_shape_{maf.shape}"
                    ),
                })

                continue

            # Compare recovered DAPI with
            # released MAF blue plane.
            ncc = normalized_cross_correlation(
                maf[..., 2],
                dapi,
            )

            # Preserve same filename.
            output_path = (
                OUTPUT_DIR
                / maf_path.name
            )

            tifffile.imwrite(
                output_path,
                dapi.astype(np.uint8),
            )

            records.append({
                "maf_filename": maf_path.name,
                "dapi_filename": output_path.name,

                "train_sample": train_sample,
                "raw_sample": raw_sample,

                "roi_x": info["roi_x"],
                "roi_y": info["roi_y"],

                "patch_x": info["patch_x"],
                "patch_y": info["patch_y"],

                "raw_x": raw_x,
                "raw_y": raw_y,

                "ncc": ncc,

                # Don't delete questionable patches yet.
                "valid": ncc >= 0.95,
                "reason": (
                    ""
                    if ncc >= 0.95
                    else "low_ncc"
                ),
            })

            if (
                i % 500 == 0
                or i == len(maf_files)
            ):

                print(
                    f"{i:>6,} / "
                    f"{len(maf_files):,}"
                )

    finally:

        for slide in slides.values():
            slide.close()

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print("\nExtraction finished.")

    print(
        "Total:",
        len(df)
    )

    print(
        "Valid:",
        df["valid"].sum()
    )

    if "ncc" in df.columns:

        print(
            "Mean NCC:",
            df["ncc"].mean()
        )

        print(
            "Median NCC:",
            df["ncc"].median()
        )

        print(
            "NCC > 0.99:",
            (df["ncc"] > 0.99).sum()
        )

        print(
            "NCC > 0.95:",
            (df["ncc"] > 0.95).sum()
        )

    print(
        "\nManifest:",
        MANIFEST_PATH
    )


if __name__ == "__main__":
    main()
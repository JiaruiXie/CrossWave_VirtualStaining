from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(0, str(ROOT))


MAF_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Training"
    / "trainA"
)

DAPI_DIR = (
    ROOT
    / "data"
    / "processed"
    / "train"
    / "dapi"
)

INPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "train_dapi_manifest.csv"
)

OUTPUT_QC = (
    ROOT
    / "data"
    / "metadata"
    / "train_pair_qc.csv"
)


# These are QC flags only.
# We will NOT automatically exclude patches
# based on intensity variance.
LOW_STD_THRESHOLD = 2.0

EXPECTED_HEIGHT = 256
EXPECTED_WIDTH = 256


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


def channel_stats(channel):

    channel = channel.astype(np.float32)

    return {
        "min": float(channel.min()),
        "max": float(channel.max()),
        "mean": float(channel.mean()),
        "std": float(channel.std()),
        "nonzero_fraction": float(
            np.count_nonzero(channel) / channel.size
        ),
    }


def main():

    manifest = pd.read_csv(
        INPUT_MANIFEST
    )

    print(
        f"Manifest rows: {len(manifest):,}"
    )

    qc_records = []

    for i, row in manifest.iterrows():

        maf_filename = row["maf_filename"]

        dapi_filename = row.get(
            "dapi_filename",
            maf_filename,
        )

        maf_path = MAF_DIR / maf_filename
        dapi_path = DAPI_DIR / dapi_filename

        record = {
            "maf_filename": maf_filename,
            "dapi_filename": dapi_filename,

            "train_sample": row.get(
                "train_sample",
                np.nan,
            ),

            "raw_sample": row.get(
                "raw_sample",
                np.nan,
            ),

            "manifest_ncc": row.get(
                "ncc",
                np.nan,
            ),

            "manifest_valid": bool(
                row.get(
                    "valid",
                    False,
                )
            ),
        }

        # --------------------------------------------------
        # File existence
        # --------------------------------------------------

        maf_exists = maf_path.exists()
        dapi_exists = dapi_path.exists()

        record["maf_exists"] = maf_exists
        record["dapi_exists"] = dapi_exists

        if not maf_exists or not dapi_exists:

            record["file_ok"] = False
            record["shape_ok"] = False
            record["qc_ncc"] = np.nan

            qc_records.append(record)
            continue

        # --------------------------------------------------
        # Load images
        # --------------------------------------------------

        try:

            maf = tifffile.imread(
                maf_path
            )

            dapi = tifffile.imread(
                dapi_path
            )

            record["file_ok"] = True

        except Exception as exc:

            record["file_ok"] = False
            record["load_error"] = str(exc)
            record["shape_ok"] = False
            record["qc_ncc"] = np.nan

            qc_records.append(record)
            continue

        # --------------------------------------------------
        # Shape checks
        # --------------------------------------------------

        record["maf_shape"] = str(
            tuple(maf.shape)
        )

        record["dapi_shape"] = str(
            tuple(dapi.shape)
        )

        maf_shape_ok = (
            maf.ndim == 3
            and maf.shape[0] == EXPECTED_HEIGHT
            and maf.shape[1] == EXPECTED_WIDTH
            and maf.shape[2] == 3
        )

        dapi_shape_ok = (
            dapi.ndim == 2
            and dapi.shape[0] == EXPECTED_HEIGHT
            and dapi.shape[1] == EXPECTED_WIDTH
        )

        record["shape_ok"] = (
            maf_shape_ok
            and dapi_shape_ok
        )

        record["maf_dtype"] = str(
            maf.dtype
        )

        record["dapi_dtype"] = str(
            dapi.dtype
        )

        if not record["shape_ok"]:

            record["qc_ncc"] = np.nan

            qc_records.append(record)
            continue

        # --------------------------------------------------
        # MAF channel statistics
        # --------------------------------------------------

        for c in range(3):

            stats = channel_stats(
                maf[..., c]
            )

            for key, value in stats.items():

                record[
                    f"maf_c{c}_{key}"
                ] = value

        # --------------------------------------------------
        # DAPI statistics
        # --------------------------------------------------

        dapi_stats = channel_stats(
            dapi
        )

        for key, value in dapi_stats.items():

            record[
                f"dapi_{key}"
            ] = value

        # --------------------------------------------------
        # Independent alignment QC
        # --------------------------------------------------

        qc_ncc = (
            normalized_cross_correlation(
                maf[..., 2],
                dapi,
            )
        )

        record["qc_ncc"] = qc_ncc

        # --------------------------------------------------
        # Informational QC flags
        # --------------------------------------------------

        maf_channel_stds = [
            record["maf_c0_std"],
            record["maf_c1_std"],
            record["maf_c2_std"],
        ]

        record["low_variance_dapi"] = (
            record["dapi_std"]
            < LOW_STD_THRESHOLD
        )

        record["low_variance_maf"] = (
            max(maf_channel_stds)
            < LOW_STD_THRESHOLD
        )

        record["maf_c0_zero"] = (
            record["maf_c0_max"] == 0
        )

        record["maf_c1_zero"] = (
            record["maf_c1_max"] == 0
        )

        record["maf_c2_zero"] = (
            record["maf_c2_max"] == 0
        )

        record["alignment_pass"] = (
            qc_ncc >= 0.95
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Do not exclude patches solely because they have
        # low variance. The original paper does not report
        # the exact background-filtering threshold.
        #
        # For now, usable means:
        #   - extraction previously passed
        #   - both files exist/load
        #   - correct shapes
        #   - alignment passes
        # --------------------------------------------------

        record["usable"] = bool(
            record["manifest_valid"]
            and record["file_ok"]
            and record["shape_ok"]
            and record["alignment_pass"]
        )

        qc_records.append(record)

        if (
            (i + 1) % 1000 == 0
            or (i + 1) == len(manifest)
        ):

            print(
                f"{i + 1:>6,} / "
                f"{len(manifest):,}"
            )

    # ------------------------------------------------------
    # Save QC table
    # ------------------------------------------------------

    qc = pd.DataFrame(
        qc_records
    )

    qc.to_csv(
        OUTPUT_QC,
        index=False,
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING PAIR QC SUMMARY")
    print("=" * 70)

    print(
        f"Total manifest rows: "
        f"{len(qc):,}"
    )

    print(
        f"Usable pairs: "
        f"{qc['usable'].sum():,}"
    )

    print(
        f"Rejected pairs: "
        f"{(~qc['usable']).sum():,}"
    )

    print(
        f"Alignment pass (NCC >= 0.95): "
        f"{qc['alignment_pass'].fillna(False).sum():,}"
    )

    print(
        f"Low-variance DAPI flags: "
        f"{qc['low_variance_dapi'].fillna(False).sum():,}"
    )

    print(
        f"Low-variance MAF flags: "
        f"{qc['low_variance_maf'].fillna(False).sum():,}"
    )

    print("\nMAF zero-channel counts:")

    for c in range(3):

        column = f"maf_c{c}_zero"

        print(
            f"  Channel {c}: "
            f"{qc[column].fillna(False).sum():,}"
        )

    valid_ncc = qc.loc[
        qc["qc_ncc"].notna(),
        "qc_ncc",
    ]

    print("\nNCC statistics:")

    print(
        valid_ncc.describe()
    )

    # ------------------------------------------------------
    # Per-sample summary
    # ------------------------------------------------------

    print("\nPer-sample QC:")

    sample_summary = (
        qc
        .groupby("train_sample")
        .agg(
            total=("maf_filename", "count"),
            usable=("usable", "sum"),
            median_ncc=("qc_ncc", "median"),
            mean_ncc=("qc_ncc", "mean"),
            low_dapi=(
                "low_variance_dapi",
                "sum",
            ),
        )
    )

    print(
        sample_summary.to_string()
    )

    # ------------------------------------------------------
    # Show worst validly loaded alignments
    # ------------------------------------------------------

    print("\nLowest NCC pairs:")

    columns = [
        "maf_filename",
        "train_sample",
        "qc_ncc",
        "dapi_std",
        "maf_c0_std",
        "maf_c1_std",
        "maf_c2_std",
    ]

    print(
        qc[
            qc["qc_ncc"].notna()
        ]
        .sort_values(
            "qc_ncc",
            ascending=True,
        )[columns]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nQC table saved to:"
    )

    print(
        OUTPUT_QC
    )


if __name__ == "__main__":
    main()
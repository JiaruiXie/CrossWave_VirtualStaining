from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import tifffile


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(0, str(ROOT))


# =========================================================
# Paths
# =========================================================

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

HE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Training"
    / "trainB"
)

QC_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "train_pair_qc.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "data_split.csv"
)

TRAIN_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "train_split.csv"
)

VAL_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "val_split.csv"
)


VAL_FRACTION = 0.10


# =========================================================
# Filename utilities
# =========================================================

def get_source_roi(filename):
    """
    Remove patch coordinates from filename.

    Example
    -------
    source.tif_y00448_x00672.tif

    becomes

    source.tif
    """

    return re.sub(
        r"_y\d+_x\d+\.tif$",
        "",
        filename,
        flags=re.IGNORECASE,
    )


def get_sample_number(filename):

    match = re.search(
        r"sample\s*(\d+)",
        filename,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return -1


# =========================================================
# ROI-level subset selection
# =========================================================

def select_validation_rois(
    df,
    group_col="source_roi",
    sample_col="sample",
    val_fraction=0.10,
):
    """
    Select whole ROIs for validation while:

    1. keeping validation close to requested patch fraction
    2. avoiding ROI leakage
    3. maximizing biological-sample diversity
    4. ensuring each represented sample still has >=1 ROI in training

    Uses dynamic programming over:
        (patch_count, sample_mask)

    Because there are only 9 breast samples, the sample mask
    is compact and computationally manageable.
    """

    group_table = (
        df
        .groupby([group_col, sample_col])
        .size()
        .reset_index(name="n_patches")
    )

    total_patches = len(df)

    target = int(
        round(total_patches * val_fraction)
    )

    # Allow a very small deviation around 10%.
    tolerance = max(
        20,
        int(round(total_patches * 0.01))
    )

    max_sum = target + tolerance

    print(
        f"Target validation patches: "
        f"{target:,} / {total_patches:,} "
        f"({val_fraction:.1%})"
    )

    print(
        f"Allowed search range: "
        f"{target - tolerance:,} "
        f"to {target + tolerance:,}"
    )

    # -----------------------------------------------------
    # Sample IDs -> bit positions
    # -----------------------------------------------------

    samples = sorted(
        group_table[sample_col].unique()
    )

    sample_to_bit = {
        sample: i
        for i, sample in enumerate(samples)
    }

    # Number of ROIs available for each sample
    roi_count_per_sample = (
        group_table
        .groupby(sample_col)
        .size()
        .to_dict()
    )

    groups = group_table.to_dict("records")

    # -----------------------------------------------------
    # Dynamic programming
    #
    # key:
    #   (patch_sum, sample_mask)
    #
    # value:
    #   integer bitset indicating selected ROI indices
    # -----------------------------------------------------

    dp = {
        (0, 0): 0
    }

    for idx, group in enumerate(groups):

        size = int(
            group["n_patches"]
        )

        sample = group[sample_col]

        sample_bit = (
            1 << sample_to_bit[sample]
        )

        current_states = list(
            dp.items()
        )

        for (
            patch_sum,
            sample_mask
        ), selected_bits in current_states:

            new_sum = (
                patch_sum + size
            )

            if new_sum > max_sum:
                continue

            new_mask = (
                sample_mask | sample_bit
            )

            key = (
                new_sum,
                new_mask
            )

            # One solution per state is enough.
            if key not in dp:

                dp[key] = (
                    selected_bits
                    | (1 << idx)
                )

    # -----------------------------------------------------
    # Evaluate solutions
    # -----------------------------------------------------

    candidates = []

    min_sum = max(
        1,
        target - tolerance
    )

    for (
        patch_sum,
        sample_mask
    ), selected_bits in dp.items():

        if (
            patch_sum < min_sum
            or patch_sum > max_sum
        ):
            continue

        selected_indices = [
            i
            for i in range(len(groups))
            if selected_bits & (1 << i)
        ]

        if not selected_indices:
            continue

        # Count how many ROIs from each sample
        # are placed into validation.
        selected_per_sample = {}

        for i in selected_indices:

            sample = (
                groups[i][sample_col]
            )

            selected_per_sample[sample] = (
                selected_per_sample.get(
                    sample,
                    0,
                )
                + 1
            )

        # Every sample represented in validation
        # must retain at least one ROI in training.
        invalid = False

        for (
            sample,
            selected_count
        ) in selected_per_sample.items():

            if (
                selected_count
                >= roi_count_per_sample[sample]
            ):

                invalid = True
                break

        if invalid:
            continue

        sample_diversity = (
            sample_mask.bit_count()
        )

        patch_error = abs(
            patch_sum - target
        )

        # -------------------------------------------------
        # Objective:
        #
        # 1. maximize biological sample diversity
        # 2. then minimize deviation from target
        # 3. then prefer fewer validation ROIs
        # -------------------------------------------------

        candidates.append(
            (
                -sample_diversity,
                patch_error,
                len(selected_indices),
                patch_sum,
                selected_indices,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Could not construct a valid "
            "diverse ROI validation split."
        )

    candidates.sort()

    (
        neg_diversity,
        patch_error,
        _,
        selected_patch_count,
        selected_indices,
    ) = candidates[0]

    selected_rois = {
        groups[i][group_col]
        for i in selected_indices
    }

    selected_samples = sorted({
        groups[i][sample_col]
        for i in selected_indices
    })

    print(
        f"Selected validation patches: "
        f"{selected_patch_count:,}"
    )

    print(
        f"Actual validation fraction: "
        f"{selected_patch_count / total_patches:.2%}"
    )

    print(
        f"Patch-count error: "
        f"{patch_error:,}"
    )

    print(
        f"Validation ROIs: "
        f"{len(selected_rois)}"
    )

    print(
        f"Validation samples "
        f"({len(selected_samples)}): "
        f"{selected_samples}"
    )

    return selected_rois


def assign_split(
    df,
    val_fraction=0.10,
):

    df = df.copy()

    val_rois = (
        select_validation_rois(
            df,
            group_col="source_roi",
            sample_col="sample",
            val_fraction=val_fraction,
        )
    )

    df["split"] = np.where(
        df["source_roi"].isin(
            val_rois
        ),
        "val",
        "train",
    )

    return df


# =========================================================
# MAF / DAPI table
# =========================================================

def build_maf_table():

    qc = pd.read_csv(
        QC_PATH
    )

    qc = qc[
        qc["usable"] == True
    ].copy()

    rows = []

    for _, row in qc.iterrows():

        filename = (
            row["maf_filename"]
        )

        rows.append({

            "domain": "maf",

            "filename": filename,

            "maf_path": str(
                MAF_DIR / filename
            ),

            "dapi_path": str(
                DAPI_DIR
                / row["dapi_filename"]
            ),

            "sample": int(
                row["train_sample"]
            ),

            "source_roi": (
                get_source_roi(
                    filename
                )
            ),

            "ncc": float(
                row["qc_ncc"]
            ),
        })

    df = pd.DataFrame(
        rows
    )

    print("\n" + "=" * 72)
    print("CREATING MAF/DAPI SPLIT")
    print("=" * 72)

    return assign_split(
        df,
        VAL_FRACTION,
    )


# =========================================================
# H&E table
# =========================================================

def build_he_table():

    he_files = sorted(
        HE_DIR.glob("*.tif")
    )

    rows = []

    print(
        f"\nChecking "
        f"{len(he_files):,} H&E patches..."
    )

    for i, path in enumerate(
        he_files,
        start=1,
    ):

        try:

            image = (
                tifffile.imread(
                    path
                )
            )

            shape_ok = (
                image.ndim == 3
                and image.shape[0] == 256
                and image.shape[1] == 256
                and image.shape[2] == 3
            )

        except Exception:
            shape_ok = False

        if not shape_ok:

            print(
                "Skipping invalid H&E:",
                path.name,
            )

            continue

        rows.append({

            "domain": "he",

            "filename": path.name,

            "he_path": str(
                path
            ),

            "sample": (
                get_sample_number(
                    path.name
                )
            ),

            "source_roi": (
                get_source_roi(
                    path.name
                )
            ),
        })

        if (
            i % 2000 == 0
            or i == len(he_files)
        ):

            print(
                f"{i:>6,} / "
                f"{len(he_files):,}"
            )

    df = pd.DataFrame(
        rows
    )

    print("\n" + "=" * 72)
    print("CREATING H&E SPLIT")
    print("=" * 72)

    return assign_split(
        df,
        VAL_FRACTION,
    )


# =========================================================
# Leakage verification
# =========================================================

def check_roi_leakage(
    df,
    domain,
):

    train_groups = set(

        df.loc[
            df["split"] == "train",
            "source_roi",
        ]

    )

    val_groups = set(

        df.loc[
            df["split"] == "val",
            "source_roi",
        ]

    )

    overlap = (
        train_groups
        & val_groups
    )

    print(
        f"{domain} ROI leakage:",
        len(overlap),
    )

    if overlap:

        raise RuntimeError(
            f"{domain}: "
            f"ROI leakage detected."
        )


# =========================================================
# Summary
# =========================================================

def summarize_domain(
    name,
    df,
):

    total = len(df)

    train_n = (
        df["split"]
        .eq("train")
        .sum()
    )

    val_n = (
        df["split"]
        .eq("val")
        .sum()
    )

    print(
        f"\n{name}"
    )

    print(
        f"Total: {total:,}"
    )

    print(
        f"Train: {train_n:,} "
        f"({train_n / total:.2%})"
    )

    print(
        f"Val:   {val_n:,} "
        f"({val_n / total:.2%})"
    )

    print(
        "Source ROIs:",
        df["source_roi"].nunique(),
    )

    print(
        "Train ROIs:",
        df.loc[
            df["split"] == "train",
            "source_roi",
        ].nunique(),
    )

    print(
        "Val ROIs:",
        df.loc[
            df["split"] == "val",
            "source_roi",
        ].nunique(),
    )

    print(
        "\nPer-sample patch counts:"
    )

    summary = (

        df
        .groupby(
            ["sample", "split"]
        )
        .size()
        .unstack(
            fill_value=0
        )

    )

    print(
        summary.to_string()
    )

    print(
        "\nValidation ROIs:"
    )

    val_roi_summary = (

        df[
            df["split"] == "val"
        ]
        .groupby(
            [
                "sample",
                "source_roi",
            ]
        )
        .size()
        .reset_index(
            name="patches"
        )

    )

    print(
        val_roi_summary.to_string(
            index=False
        )
    )


# =========================================================
# Main
# =========================================================

def main():

    maf_df = (
        build_maf_table()
    )

    he_df = (
        build_he_table()
    )

    check_roi_leakage(
        maf_df,
        "MAF",
    )

    check_roi_leakage(
        he_df,
        "H&E",
    )

    combined = pd.concat(
        [
            maf_df,
            he_df,
        ],
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    combined[
        combined["split"]
        == "train"
    ].to_csv(
        TRAIN_PATH,
        index=False,
    )

    combined[
        combined["split"]
        == "val"
    ].to_csv(
        VAL_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL DATA SPLIT SUMMARY"
    )

    print(
        "=" * 72
    )

    summarize_domain(
        "MAF/DAPI",
        maf_df,
    )

    summarize_domain(
        "H&E",
        he_df,
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        TRAIN_PATH
    )

    print(
        VAL_PATH
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp


Image.MAX_IMAGE_PIXELS = None


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]

PAIR_MANIFEST = (
    ROOT
    / "outputs"
    / "comparison_figures"
    / "separate_pairs"
    / "generated_pair_comparisons.csv"
)

CROSSWAVE_DIR = (
    ROOT
    / "outputs"
    / "crosswave_dapi_student_lr_v3"
    / "sub_eval_vhe"
)

GT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Sub_eval"
    / "GroudTruth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "morphology_crosswave_vs_gt_fast"
    / "cell_features_fast"
)


FEATURES = [
    "cell_area",
    "cell_eccentricity",
    "cell_circularity",
    "nucleus_cell_area_ratio",
]


# ============================================================
# I/O
# ============================================================

def save_csv(rows, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def read_pairs():

    if not PAIR_MANIFEST.is_file():

        raise FileNotFoundError(
            f"Pair manifest not found:\n{PAIR_MANIFEST}"
        )

    with PAIR_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(
                handle
            )
        )

    pairs = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        cw_name = row.get(
            "crosswave",
            "",
        ).strip()

        gt_name = row.get(
            "ground_truth",
            "",
        ).strip()

        if not cw_name or not gt_name:
            continue

        cw_path = (
            CROSSWAVE_DIR
            / cw_name
        )

        gt_path = (
            GT_DIR
            / gt_name
        )

        if not cw_path.is_file():
            raise FileNotFoundError(
                cw_path
            )

        if not gt_path.is_file():
            raise FileNotFoundError(
                gt_path
            )

        pairs.append(
            {
                "roi_id":
                    f"roi_{index:03d}",

                "crosswave":
                    cw_path,

                "ground_truth":
                    gt_path,
            }
        )

    return pairs


# ============================================================
# IMAGE LOADING
# ============================================================

def load_rgb(
    path,
    scale,
):

    with Image.open(
        path
    ) as image:

        image = image.convert(
            "RGB"
        )

        if scale != 1.0:

            new_size = (
                max(
                    1,
                    int(
                        round(
                            image.width
                            * scale
                        )
                    ),
                ),
                max(
                    1,
                    int(
                        round(
                            image.height
                            * scale
                        )
                    ),
                ),
            )

            image = image.resize(
                new_size,
                Image.Resampling.BILINEAR,
            )

        return np.asarray(
            image,
            dtype=np.uint8,
        )


# ============================================================
# HEMATOXYLIN CHANNEL
# ============================================================

def hematoxylin_channel(
    rgb,
):

    rgb_float = np.clip(
        rgb.astype(
            np.float32
        )
        / 255.0,
        1.0 / 255.0,
        1.0,
    )

    od = -np.log(
        rgb_float
    )

    r = od[:, :, 0]
    g = od[:, :, 1]
    b = od[:, :, 2]

    hematoxylin = (
        1.88000259 * r
        - 0.07359820 * g
        - 0.59521871 * b
    )

    return np.maximum(
        hematoxylin,
        0.0,
    ).astype(
        np.float32
    )


def robust_uint8(
    image,
):

    finite = image[
        np.isfinite(
            image
        )
    ]

    if finite.size == 0:

        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    lo = float(
        np.percentile(
            finite,
            1.0,
        )
    )

    hi = float(
        np.percentile(
            finite,
            99.5,
        )
    )

    if hi <= lo:

        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    normalized = np.clip(
        (
            image - lo
        )
        / (
            hi - lo
        ),
        0.0,
        1.0,
    )

    return np.rint(
        normalized
        * 255.0
    ).astype(
        np.uint8
    )


# ============================================================
# FAST NUCLEUS SEGMENTATION
# ============================================================

def segment_nuclei_fast(
    rgb,
    min_area,
    max_area,
    peak_fraction,
):

    hematoxylin = (
        hematoxylin_channel(
            rgb
        )
    )

    h8 = robust_uint8(
        hematoxylin
    )

    h8 = cv2.GaussianBlur(
        h8,
        (3, 3),
        0,
    )

    otsu, binary = cv2.threshold(
        h8,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    (
        count,
        cc_labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        (
            binary > 0
        ).astype(
            np.uint8
        ),
        connectivity=8,
    )

    filtered = np.zeros_like(
        binary,
        dtype=np.uint8,
    )

    for label in range(
        1,
        count,
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= min_area:

            filtered[
                cc_labels == label
            ] = 1

    distance = cv2.distanceTransform(
        filtered,
        cv2.DIST_L2,
        5,
    )

    if float(
        distance.max()
    ) <= 0:

        return np.zeros_like(
            filtered,
            dtype=np.int32,
        )

    local_max = (
        distance
        == ndi.maximum_filter(
            distance,
            size=5,
            mode="nearest",
        )
    )

    local_max &= (
        distance
        >= peak_fraction
        * float(
            distance.max()
        )
    )

    markers, n_markers = ndi.label(
        local_max
    )

    if n_markers == 0:

        markers, _ = ndi.label(
            filtered
        )

    markers = markers.astype(
        np.int32
    )

    watershed_input = cv2.cvtColor(
        255 - h8,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.watershed(
        watershed_input,
        markers,
    )

    output = np.zeros_like(
        markers,
        dtype=np.int32,
    )

    next_id = 1

    for old_id in np.unique(
        markers
    ):

        if old_id <= 0:
            continue

        mask = (
            markers == old_id
        ) & (
            filtered > 0
        )

        area = int(
            mask.sum()
        )

        if (
            min_area
            <= area
            <= max_area
        ):

            output[
                mask
            ] = next_id

            next_id += 1

    return output


# ============================================================
# FAST TISSUE MASK
# ============================================================

def tissue_mask(
    rgb,
):

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    tissue = (
        gray < 245
    ).astype(
        np.uint8
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    tissue = cv2.morphologyEx(
        tissue,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return tissue.astype(
        bool
    )


# ============================================================
# FAST APPROXIMATE CELL ASSIGNMENT
# ============================================================

def approximate_cell_labels_fast(
    rgb,
    nucleus_labels,
    max_radius,
):

    nucleus_mask = (
        nucleus_labels > 0
    )

    if not np.any(
        nucleus_mask
    ):

        return np.zeros_like(
            nucleus_labels,
            dtype=np.int32,
        )

    non_nucleus = (
        ~nucleus_mask
    )

    (
        distances,
        nearest_indices,
    ) = ndi.distance_transform_edt(
        non_nucleus,
        return_indices=True,
    )

    nearest_labels = nucleus_labels[
        nearest_indices[0],
        nearest_indices[1],
    ]

    cells = nearest_labels.astype(
        np.int32
    )

    valid = (
        tissue_mask(
            rgb
        )
        & (
            distances <= max_radius
        )
    )

    cells[
        ~valid
    ] = 0

    cells[
        nucleus_mask
    ] = nucleus_labels[
        nucleus_mask
    ]

    return cells


# ============================================================
# SHAPE HELPERS
# ============================================================

def largest_contour(
    mask,
):

    contours, _ = cv2.findContours(
        mask.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea,
    )


def eccentricity_from_contour(
    contour,
):

    if (
        contour is None
        or len(
            contour
        ) < 5
    ):
        return np.nan

    try:

        ellipse = cv2.fitEllipse(
            contour
        )

        major = max(
            ellipse[1]
        )

        minor = min(
            ellipse[1]
        )

        if major <= 0:
            return 0.0

        return math.sqrt(
            max(
                0.0,
                1.0
                - (
                    minor * minor
                )
                / (
                    major * major
                ),
            )
        )

    except cv2.error:

        return np.nan


# ============================================================
# EXTRACT ONLY THE 4 MISSING FEATURES
# ============================================================

def extract_cell_features(
    nucleus_labels,
    cell_labels,
    roi_id,
    domain,
    scale,
    min_cell_area_original,
):

    rows = []

    area_factor = (
        1.0
        / (
            scale
            * scale
        )
    )

    max_id = int(
        nucleus_labels.max()
    )

    for object_id in range(
        1,
        max_id + 1,
    ):

        nucleus_mask = (
            nucleus_labels
            == object_id
        )

        nucleus_area_scaled = int(
            nucleus_mask.sum()
        )

        if nucleus_area_scaled <= 0:
            continue

        cell_mask = (
            cell_labels
            == object_id
        ).astype(
            np.uint8
        )

        cell_area_scaled = int(
            cell_mask.sum()
        )

        cell_area_original = (
            cell_area_scaled
            * area_factor
        )

        if (
            cell_area_original
            < min_cell_area_original
        ):
            continue

        contour = largest_contour(
            cell_mask
        )

        if contour is None:
            continue

        contour_area_scaled = float(
            cv2.contourArea(
                contour
            )
        )

        perimeter_scaled = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        # Use contour geometry for circularity
        if (
            contour_area_scaled > 0
            and perimeter_scaled > 0
        ):

            cell_circularity = (
                4.0
                * math.pi
                * contour_area_scaled
                / (
                    perimeter_scaled
                    ** 2
                )
            )

        else:

            cell_circularity = np.nan

        cell_eccentricity = (
            eccentricity_from_contour(
                contour
            )
        )

        nucleus_area_original = (
            nucleus_area_scaled
            * area_factor
        )

        nucleus_cell_ratio = (
            nucleus_area_original
            / cell_area_original
            if cell_area_original > 0
            else np.nan
        )

        rows.append(
            {
                "domain":
                    domain,

                "roi_id":
                    roi_id,

                "object_id":
                    object_id,

                "cell_area":
                    cell_area_original,

                "cell_eccentricity":
                    cell_eccentricity,

                "cell_circularity":
                    cell_circularity,

                "nucleus_cell_area_ratio":
                    nucleus_cell_ratio,
            }
        )

    return rows


# ============================================================
# DISTRIBUTION METRICS
# ============================================================

def finite_values(
    rows,
    feature,
):

    values = []

    for row in rows:

        try:

            value = float(
                row[
                    feature
                ]
            )

        except Exception:

            continue

        if math.isfinite(
            value
        ):

            values.append(
                value
            )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def cohens_d(
    x,
    y,
):

    if (
        x.size < 2
        or y.size < 2
    ):
        return np.nan

    pooled_var = (
        (
            x.size - 1
        )
        * x.var(
            ddof=1
        )
        + (
            y.size - 1
        )
        * y.var(
            ddof=1
        )
    ) / (
        x.size
        + y.size
        - 2
    )

    if pooled_var <= 0:
        return 0.0

    return float(
        (
            x.mean()
            - y.mean()
        )
        / math.sqrt(
            pooled_var
        )
    )


def compare_distributions(
    cw_rows,
    gt_rows,
    feature,
    bins,
):

    x = finite_values(
        cw_rows,
        feature,
    )

    y = finite_values(
        gt_rows,
        feature,
    )

    out = {
        "feature":
            feature,

        "n_crosswave":
            int(
                x.size
            ),

        "n_ground_truth":
            int(
                y.size
            ),

        "mean_crosswave":
            float(
                x.mean()
            )
            if x.size
            else np.nan,

        "mean_ground_truth":
            float(
                y.mean()
            )
            if y.size
            else np.nan,

        "overlap_coefficient":
            np.nan,

        "js_divergence":
            np.nan,

        "ks_statistic":
            np.nan,

        "ks_pvalue":
            np.nan,

        "cohens_d":
            np.nan,

        "abs_cohens_d":
            np.nan,
    }

    if (
        x.size == 0
        or y.size == 0
    ):
        return out

    combined = np.concatenate(
        [
            x,
            y,
        ]
    )

    lo = float(
        np.percentile(
            combined,
            0.5,
        )
    )

    hi = float(
        np.percentile(
            combined,
            99.5,
        )
    )

    if hi <= lo:
        hi = lo + 1e-6

    hx, edges = np.histogram(
        x,
        bins=bins,
        range=(
            lo,
            hi,
        ),
    )

    hy, _ = np.histogram(
        y,
        bins=edges,
    )

    p = hx.astype(
        np.float64
    )

    q = hy.astype(
        np.float64
    )

    if (
        p.sum() > 0
        and q.sum() > 0
    ):

        p /= p.sum()
        q /= q.sum()

        out[
            "overlap_coefficient"
        ] = float(
            np.minimum(
                p,
                q,
            ).sum()
        )

        eps = 1e-12

        p2 = np.clip(
            p,
            eps,
            None,
        )

        q2 = np.clip(
            q,
            eps,
            None,
        )

        p2 /= p2.sum()
        q2 /= q2.sum()

        m = 0.5 * (
            p2
            + q2
        )

        out[
            "js_divergence"
        ] = float(
            0.5
            * np.sum(
                p2
                * np.log(
                    p2 / m
                )
            )
            + 0.5
            * np.sum(
                q2
                * np.log(
                    q2 / m
                )
            )
        )

    ks = ks_2samp(
        x,
        y,
        alternative="two-sided",
        method="auto",
    )

    out[
        "ks_statistic"
    ] = float(
        ks.statistic
    )

    out[
        "ks_pvalue"
    ] = float(
        ks.pvalue
    )

    d = cohens_d(
        x,
        y,
    )

    out[
        "cohens_d"
    ] = d

    out[
        "abs_cohens_d"
    ] = (
        abs(
            d
        )
        if math.isfinite(
            d
        )
        else np.nan
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--min-nucleus-area",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--max-nucleus-area",
        type=float,
        default=2500.0,
    )

    parser.add_argument(
        "--min-cell-area",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--max-cell-radius",
        type=float,
        default=18.0,
        help=(
            "Cell radius in ORIGINAL-resolution pixels."
        ),
    )

    parser.add_argument(
        "--watershed-peak-fraction",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    min_area_scaled = max(
        3,
        int(
            round(
                args.min_nucleus_area
                * args.scale
                * args.scale
            )
        ),
    )

    max_area_scaled = max(
        min_area_scaled + 1,
        int(
            round(
                args.max_nucleus_area
                * args.scale
                * args.scale
            )
        ),
    )

    max_cell_radius_scaled = (
        args.max_cell_radius
        * args.scale
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = read_pairs()

    if args.max_pairs > 0:

        pairs = pairs[
            :args.max_pairs
        ]

    print()
    print("=" * 78)
    print("FAST CELL FEATURES ONLY")
    print("=" * 78)

    print(
        f"Pairs: {len(pairs)}"
    )

    print(
        f"Scale: {args.scale}"
    )

    print(
        f"Max cell radius at scaled resolution: "
        f"{max_cell_radius_scaled:.2f} px"
    )

    print(
        "Features:"
    )

    for feature in FEATURES:
        print(
            " ",
            feature,
        )

    print()

    cw_rows = []
    gt_rows = []

    run_start = time.time()

    for pair_index, pair in enumerate(
        pairs,
        start=1,
    ):

        pair_start = time.time()

        print(
            "-" * 78,
            flush=True,
        )

        print(
            f"PAIR {pair_index}/{len(pairs)} "
            f"| {pair['roi_id']}",
            flush=True,
        )

        for domain, path in (
            (
                "crosswave",
                pair[
                    "crosswave"
                ],
            ),
            (
                "ground_truth",
                pair[
                    "ground_truth"
                ],
            ),
        ):

            t0 = time.time()

            rgb = load_rgb(
                path,
                args.scale,
            )

            labels = (
                segment_nuclei_fast(
                    rgb,
                    min_area_scaled,
                    max_area_scaled,
                    args.watershed_peak_fraction,
                )
            )

            segmentation_seconds = (
                time.time()
                - t0
            )

            t0 = time.time()

            cells = (
                approximate_cell_labels_fast(
                    rgb,
                    labels,
                    max_radius=
                        max_cell_radius_scaled,
                )
            )

            cell_assignment_seconds = (
                time.time()
                - t0
            )

            t0 = time.time()

            rows = extract_cell_features(
                nucleus_labels=
                    labels,

                cell_labels=
                    cells,

                roi_id=
                    pair[
                        "roi_id"
                    ],

                domain=
                    domain,

                scale=
                    args.scale,

                min_cell_area_original=
                    args.min_cell_area,
            )

            feature_seconds = (
                time.time()
                - t0
            )

            print(
                f"  {domain:13s} "
                f"objects={len(rows):,} | "
                f"segment={segmentation_seconds:.1f}s | "
                f"cells={cell_assignment_seconds:.1f}s | "
                f"features={feature_seconds:.1f}s",
                flush=True,
            )

            if domain == "crosswave":

                cw_rows.extend(
                    rows
                )

            else:

                gt_rows.extend(
                    rows
                )

        save_csv(
            cw_rows,
            OUTPUT_DIR
            / "crosswave_cell_features_partial.csv",
        )

        save_csv(
            gt_rows,
            OUTPUT_DIR
            / "ground_truth_cell_features_partial.csv",
        )

        print(
            f"  DONE in "
            f"{time.time() - pair_start:.1f}s",
            flush=True,
        )

        print()

    # ========================================================
    # SAVE OBJECT FEATURES
    # ========================================================

    save_csv(
        cw_rows,
        OUTPUT_DIR
        / "crosswave_cell_features.csv",
    )

    save_csv(
        gt_rows,
        OUTPUT_DIR
        / "ground_truth_cell_features.csv",
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    overall = [
        compare_distributions(
            cw_rows,
            gt_rows,
            feature,
            args.hist_bins,
        )
        for feature in FEATURES
    ]

    save_csv(
        overall,
        OUTPUT_DIR
        / "cell_features_metrics_overall.csv",
    )

    # ========================================================
    # PER-ROI METRICS
    # ========================================================

    per_roi = []

    for pair in pairs:

        roi_id = pair[
            "roi_id"
        ]

        cw_roi = [
            row
            for row in cw_rows
            if row[
                "roi_id"
            ] == roi_id
        ]

        gt_roi = [
            row
            for row in gt_rows
            if row[
                "roi_id"
            ] == roi_id
        ]

        for feature in FEATURES:

            result = (
                compare_distributions(
                    cw_roi,
                    gt_roi,
                    feature,
                    args.hist_bins,
                )
            )

            per_roi.append(
                {
                    "roi_id":
                        roi_id,
                    **result,
                }
            )

    save_csv(
        per_roi,
        OUTPUT_DIR
        / "cell_features_metrics_by_roi.csv",
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print("FINAL CELL-FEATURE RESULTS")
    print("=" * 78)

    for row in overall:

        print(
            f"{row['feature']:28s} "
            f"overlap={row['overlap_coefficient']:.4f} "
            f"JS={row['js_divergence']:.4f} "
            f"KS={row['ks_statistic']:.4f} "
            f"p={row['ks_pvalue']:.3g} "
            f"d={row['cohens_d']:.4f}"
        )

    print()

    print(
        f"Total runtime: "
        f"{(time.time() - run_start) / 60.0:.2f} min"
    )

    print(
        f"Saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
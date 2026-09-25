"""
Compare CrossWave virtual H&E vs ground-truth H&E morphometrics,
with visible intermediate progress and partial-result saving.

This version is designed to make long runs easier to monitor.

What is new
-----------
For every ROI pair, the script now prints:

    - which ROI is being processed
    - CrossWave image size
    - Ground-truth image size
    - nucleus segmentation time
    - number of nuclei detected
    - approximate cell-assignment time
    - feature-extraction time
    - running total number of objects
    - selected intermediate feature means

After EVERY ROI pair, it also writes partial CSV files:

    outputs/morphology_crosswave_vs_gt/
        crosswave_object_features_partial.csv
        ground_truth_object_features_partial.csv
        segmentation_summary_partial.csv

So even if the run is interrupted, you can inspect the progress already made.

It also prints an approximate ETA after the first ROI has completed.

Run
---
& "C:/Users/jiaru/.conda/envs/crosswave_vs/python.exe" `
  scripts/20_compare_crosswave_gt_morphology_v2.py

For a quick test:
& "C:/Users/jiaru/.conda/envs/crosswave_vs/python.exe" `
  scripts/20_compare_crosswave_gt_morphology_v2.py `
  --max-pairs 3
"""

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
    / "crosswave_dapi_student_newTeacher_v2"
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
    / "morphology_crosswave_vs_gt"
)

FEATURES = [
    "cell_area",
    "cell_eccentricity",
    "cell_circularity",
    "nucleus_max_caliper",
    "nucleus_perimeter",
    "nucleus_area",
    "nucleus_eccentricity",
    "nucleus_circularity",
    "nucleus_cell_area_ratio",
    "nucleus_eosin_od_mean",
]


# =====================================================================
# I/O
# =====================================================================


def read_pairs(manifest_path):
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Matched-pair manifest not found:\n"
            f"{manifest_path}\n\n"
            "Run scripts/19_plot_subeval_pairs_separately.py first."
        )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    pairs = []

    for index, row in enumerate(rows, start=1):
        cw_name = row.get("crosswave", "").strip()
        gt_name = row.get("ground_truth", "").strip()

        if not cw_name or not gt_name:
            continue

        cw_path = CROSSWAVE_DIR / cw_name
        gt_path = GT_DIR / gt_name

        if not cw_path.is_file():
            raise FileNotFoundError(
                f"CrossWave image not found: {cw_path}"
            )

        if not gt_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth image not found: {gt_path}"
            )

        pairs.append(
            {
                "roi_id": f"roi_{index:03d}",
                "crosswave": cw_path,
                "ground_truth": gt_path,
            }
        )

    return pairs


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# H&E optical-density decomposition
# =====================================================================


def he_channels(rgb):
    rgb_float = np.clip(
        rgb.astype(np.float32) / 255.0,
        1.0 / 255.0,
        1.0,
    )

    od = -np.log(rgb_float)

    r = od[:, :, 0]
    g = od[:, :, 1]
    b = od[:, :, 2]

    hematoxylin = (
        1.88000259 * r
        - 0.07359820 * g
        - 0.59521871 * b
    )

    eosin = (
        -1.01536655 * r
        + 1.13316777 * g
        - 0.48168478 * b
    )

    return (
        np.maximum(
            hematoxylin,
            0.0,
        ).astype(np.float32),
        np.maximum(
            eosin,
            0.0,
        ).astype(np.float32),
    )


def robust_uint8(image):
    finite = image[
        np.isfinite(image)
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

    scaled = np.clip(
        (image - lo)
        / (hi - lo),
        0.0,
        1.0,
    )

    return np.rint(
        scaled * 255.0
    ).astype(np.uint8)


# =====================================================================
# Segmentation
# =====================================================================


def remove_small_components(binary, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )

    output = np.zeros_like(
        binary,
        dtype=np.uint8,
    )

    for label in range(1, count):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= min_area:
            output[
                labels == label
            ] = 1

    return output


def segment_nuclei(
    rgb,
    min_area,
    max_area,
    peak_fraction,
):
    hematoxylin, eosin = he_channels(rgb)

    h8 = robust_uint8(
        hematoxylin
    )

    h8 = cv2.GaussianBlur(
        h8,
        (3, 3),
        0,
    )

    otsu_value, binary = cv2.threshold(
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

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    binary = remove_small_components(
        binary > 0,
        min_area,
    )

    distance = cv2.distanceTransform(
        binary.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    if float(distance.max()) <= 0:
        return (
            np.zeros_like(
                binary,
                dtype=np.int32,
            ),
            eosin,
            float(otsu_value),
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
        * float(distance.max())
    )

    markers, marker_count = ndi.label(
        local_max
    )

    if marker_count == 0:
        markers, _ = ndi.label(
            binary
        )

    watershed_markers = markers.astype(
        np.int32
    )

    watershed_input = cv2.cvtColor(
        255 - h8,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.watershed(
        watershed_input,
        watershed_markers,
    )

    labels_out = np.zeros_like(
        watershed_markers,
        dtype=np.int32,
    )

    next_id = 1

    for label in np.unique(
        watershed_markers
    ):
        if label <= 0:
            continue

        mask = (
            watershed_markers == label
        ) & (
            binary > 0
        )

        area = int(
            mask.sum()
        )

        if (
            min_area
            <= area
            <= max_area
        ):
            labels_out[
                mask
            ] = next_id

            next_id += 1

    return (
        labels_out,
        eosin,
        float(otsu_value),
    )


def tissue_mask(rgb):
    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    mask = (
        gray < 245
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask.astype(bool)


def approximate_cell_labels(
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
        tissue_mask(rgb)
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


# =====================================================================
# Morphometry
# =====================================================================


def largest_contour(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea,
    )


def eccentricity(mask):
    coords = np.column_stack(
        np.nonzero(mask)
    )

    if coords.shape[0] < 5:
        return np.nan

    y = coords[:, 0].astype(np.float64)
    x = coords[:, 1].astype(np.float64)

    x -= x.mean()
    y -= y.mean()

    denominator = max(
        len(x) - 1,
        1,
    )

    a = float(
        np.sum(x * x)
        / denominator
    )

    d = float(
        np.sum(y * y)
        / denominator
    )

    b = float(
        np.sum(x * y)
        / denominator
    )

    trace = a + d

    disc = math.sqrt(
        max(
            0.0,
            (a - d) ** 2
            + 4.0 * b * b,
        )
    )

    lambda_max = 0.5 * (
        trace + disc
    )

    lambda_min = 0.5 * (
        trace - disc
    )

    if lambda_max <= 0:
        return 0.0

    ratio = np.clip(
        lambda_min / lambda_max,
        0.0,
        1.0,
    )

    return math.sqrt(
        1.0 - float(ratio)
    )


def circularity(area, perimeter):
    if perimeter <= 0:
        return np.nan

    return (
        4.0
        * math.pi
        * area
        / (
            perimeter ** 2
        )
    )


def maximum_caliper(contour):
    if contour is None:
        return np.nan

    hull = cv2.convexHull(
        contour
    ).reshape(
        -1,
        2,
    ).astype(np.float32)

    if hull.shape[0] < 2:
        return 0.0

    if hull.shape[0] > 160:
        indices = np.linspace(
            0,
            hull.shape[0] - 1,
            160,
        )

        hull = hull[
            np.rint(
                indices
            ).astype(int)
        ]

    dx = (
        hull[:, 0][:, None]
        - hull[:, 0][None, :]
    )

    dy = (
        hull[:, 1][:, None]
        - hull[:, 1][None, :]
    )

    distance_sq = (
        dx * dx
        + dy * dy
    )

    return math.sqrt(
        float(
            distance_sq.max()
        )
    )


def extract_features(
    nucleus_labels,
    cell_labels,
    eosin,
    roi_id,
    domain,
    min_cell_area,
):
    rows = []

    nucleus_ids = np.unique(
        nucleus_labels
    )

    nucleus_ids = nucleus_ids[
        nucleus_ids > 0
    ]

    for object_id in nucleus_ids:
        nucleus_mask = (
            nucleus_labels
            == object_id
        )

        nucleus_area = float(
            nucleus_mask.sum()
        )

        contour = largest_contour(
            nucleus_mask
        )

        if contour is None:
            continue

        nucleus_perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        eosin_values = eosin[
            nucleus_mask
        ]

        cell_mask = (
            cell_labels
            == object_id
        )

        cell_area = float(
            cell_mask.sum()
        )

        if cell_area >= min_cell_area:
            cell_contour = largest_contour(
                cell_mask
            )

            if cell_contour is not None:
                cell_perimeter = float(
                    cv2.arcLength(
                        cell_contour,
                        True,
                    )
                )
            else:
                cell_perimeter = np.nan

            cell_ecc = eccentricity(
                cell_mask
            )

            cell_circ = circularity(
                cell_area,
                cell_perimeter,
            )

            nc_ratio = (
                nucleus_area
                / cell_area
            )
        else:
            cell_area = np.nan
            cell_ecc = np.nan
            cell_circ = np.nan
            nc_ratio = np.nan

        rows.append(
            {
                "domain":
                    domain,
                "roi_id":
                    roi_id,
                "object_id":
                    int(object_id),
                "cell_area":
                    cell_area,
                "cell_eccentricity":
                    cell_ecc,
                "cell_circularity":
                    cell_circ,
                "nucleus_max_caliper":
                    maximum_caliper(
                        contour
                    ),
                "nucleus_perimeter":
                    nucleus_perimeter,
                "nucleus_area":
                    nucleus_area,
                "nucleus_eccentricity":
                    eccentricity(
                        nucleus_mask
                    ),
                "nucleus_circularity":
                    circularity(
                        nucleus_area,
                        nucleus_perimeter,
                    ),
                "nucleus_cell_area_ratio":
                    nc_ratio,
                "nucleus_eosin_od_mean":
                    float(
                        eosin_values.mean()
                    )
                    if eosin_values.size
                    else np.nan,
            }
        )

    return rows


# =====================================================================
# Distribution statistics
# =====================================================================


def values(rows, feature):
    output = []

    for row in rows:
        try:
            value = float(
                row[feature]
            )
        except (
            TypeError,
            ValueError,
            KeyError,
        ):
            continue

        if math.isfinite(value):
            output.append(value)

    return np.asarray(
        output,
        dtype=np.float64,
    )


def cohens_d(x, y):
    if x.size < 2 or y.size < 2:
        return np.nan

    variance = (
        (
            x.size - 1
        )
        * x.var(ddof=1)
        + (
            y.size - 1
        )
        * y.var(ddof=1)
    ) / (
        x.size
        + y.size
        - 2
    )

    if variance <= 0:
        return 0.0

    return float(
        (
            x.mean()
            - y.mean()
        )
        / math.sqrt(
            float(variance)
        )
    )


def compare_distributions(
    cw_rows,
    gt_rows,
    feature,
    bins,
):
    x = values(
        cw_rows,
        feature,
    )

    y = values(
        gt_rows,
        feature,
    )

    result = {
        "feature":
            feature,
        "n_crosswave":
            int(x.size),
        "n_ground_truth":
            int(y.size),
        "mean_crosswave":
            float(x.mean())
            if x.size
            else np.nan,
        "mean_ground_truth":
            float(y.mean())
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

    if x.size == 0 or y.size == 0:
        return result

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

    p = hx.astype(np.float64)
    q = hy.astype(np.float64)

    if p.sum() > 0 and q.sum() > 0:
        p /= p.sum()
        q /= q.sum()

        result[
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
            p2 + q2
        )

        result[
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

    result[
        "ks_statistic"
    ] = float(
        ks.statistic
    )

    result[
        "ks_pvalue"
    ] = float(
        ks.pvalue
    )

    d = cohens_d(
        x,
        y,
    )

    result[
        "cohens_d"
    ] = d

    result[
        "abs_cohens_d"
    ] = (
        abs(d)
        if math.isfinite(d)
        else np.nan
    )

    return result


# =====================================================================
# Progress helpers
# =====================================================================


def safe_mean(
    rows,
    feature,
):
    arr = values(
        rows,
        feature,
    )

    if arr.size == 0:
        return np.nan

    return float(
        arr.mean()
    )


def print_domain_intermediate(
    domain,
    rows,
    otsu,
    segment_seconds,
    cell_seconds,
    feature_seconds,
):
    print(
        f"    {domain}:",
        flush=True,
    )

    print(
        f"      nuclei detected: {len(rows)}",
        flush=True,
    )

    print(
        f"      Otsu threshold: {otsu:.1f}",
        flush=True,
    )

    print(
        (
            f"      timing: segmentation={segment_seconds:.1f}s, "
            f"cells={cell_seconds:.1f}s, "
            f"features={feature_seconds:.1f}s"
        ),
        flush=True,
    )

    if rows:
        mean_area = safe_mean(
            rows,
            "nucleus_area",
        )

        mean_perimeter = safe_mean(
            rows,
            "nucleus_perimeter",
        )

        mean_ecc = safe_mean(
            rows,
            "nucleus_eccentricity",
        )

        print(
            (
                f"      intermediate means: "
                f"nucleus area={mean_area:.2f}, "
                f"perimeter={mean_perimeter:.2f}, "
                f"eccentricity={mean_ecc:.3f}"
            ),
            flush=True,
        )


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-nucleus-area",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-nucleus-area",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--min-cell-area",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-cell-radius",
        type=float,
        default=18.0,
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
        help=(
            "0 = use every matched pair."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 78,
        flush=True,
    )

    print(
        "CrossWave vs Ground Truth Morphology",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        "Step 1/4: reading matched pairs...",
        flush=True,
    )

    pairs = read_pairs(
        PAIR_MANIFEST
    )

    if args.max_pairs > 0:
        pairs = pairs[
            :args.max_pairs
        ]

    print(
        f"Pairs to process: {len(pairs)}",
        flush=True,
    )

    crosswave_rows = []
    ground_truth_rows = []
    segmentation_summary = []

    run_start = time.time()
    completed_pair_times = []

    print()
    print(
        "Step 2/4: segmenting and extracting morphometrics...",
        flush=True,
    )

    for index, pair in enumerate(
        pairs,
        start=1,
    ):
        pair_start = time.time()

        print()
        print(
            "-" * 78,
            flush=True,
        )

        print(
            (
                f"PAIR {index}/{len(pairs)} "
                f"| {pair['roi_id']}"
            ),
            flush=True,
        )

        print(
            f"  CrossWave: {pair['crosswave'].name}",
            flush=True,
        )

        print(
            f"  Ground truth: {pair['ground_truth'].name}",
            flush=True,
        )

        pair_counts = {
            "roi_id":
                pair[
                    "roi_id"
                ],
            "crosswave_file":
                pair[
                    "crosswave"
                ].name,
            "ground_truth_file":
                pair[
                    "ground_truth"
                ].name,
        }

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
            rgb = load_rgb(
                path
            )

            print(
                (
                    f"    {domain} image size: "
                    f"{rgb.shape[1]} x {rgb.shape[0]}"
                ),
                flush=True,
            )

            t0 = time.time()

            (
                nuclei,
                eosin,
                otsu,
            ) = segment_nuclei(
                rgb,
                min_area=
                    args.min_nucleus_area,
                max_area=
                    args.max_nucleus_area,
                peak_fraction=
                    args.watershed_peak_fraction,
            )

            segment_seconds = (
                time.time()
                - t0
            )

            t0 = time.time()

            cells = approximate_cell_labels(
                rgb,
                nuclei,
                max_radius=
                    args.max_cell_radius,
            )

            cell_seconds = (
                time.time()
                - t0
            )

            t0 = time.time()

            object_rows = extract_features(
                nucleus_labels=
                    nuclei,
                cell_labels=
                    cells,
                eosin=
                    eosin,
                roi_id=
                    pair[
                        "roi_id"
                    ],
                domain=
                    domain,
                min_cell_area=
                    args.min_cell_area,
            )

            feature_seconds = (
                time.time()
                - t0
            )

            print_domain_intermediate(
                domain=
                    domain,
                rows=
                    object_rows,
                otsu=
                    otsu,
                segment_seconds=
                    segment_seconds,
                cell_seconds=
                    cell_seconds,
                feature_seconds=
                    feature_seconds,
            )

            pair_counts[
                f"{domain}_nuclei"
            ] = len(
                object_rows
            )

            pair_counts[
                f"{domain}_otsu"
            ] = otsu

            pair_counts[
                f"{domain}_segmentation_seconds"
            ] = segment_seconds

            pair_counts[
                f"{domain}_cell_seconds"
            ] = cell_seconds

            pair_counts[
                f"{domain}_feature_seconds"
            ] = feature_seconds

            if domain == "crosswave":
                crosswave_rows.extend(
                    object_rows
                )
            else:
                ground_truth_rows.extend(
                    object_rows
                )

        segmentation_summary.append(
            pair_counts
        )

        # ---------------------------------------------------------
        # SAVE INTERMEDIATE RESULTS AFTER EVERY PAIR
        # ---------------------------------------------------------

        save_csv(
            crosswave_rows,
            OUTPUT_DIR
            / "crosswave_object_features_partial.csv",
        )

        save_csv(
            ground_truth_rows,
            OUTPUT_DIR
            / "ground_truth_object_features_partial.csv",
        )

        save_csv(
            segmentation_summary,
            OUTPUT_DIR
            / "segmentation_summary_partial.csv",
        )

        pair_seconds = (
            time.time()
            - pair_start
        )

        completed_pair_times.append(
            pair_seconds
        )

        average_pair_seconds = sum(
            completed_pair_times
        ) / len(
            completed_pair_times
        )

        remaining_pairs = (
            len(
                pairs
            )
            - index
        )

        eta_seconds = (
            average_pair_seconds
            * remaining_pairs
        )

        print()
        print(
            (
                f"  Pair complete in {pair_seconds:.1f}s "
                f"({pair_seconds / 60.0:.1f} min)"
            ),
            flush=True,
        )

        print(
            (
                f"  Running object totals: "
                f"CrossWave={len(crosswave_rows)}, "
                f"GT={len(ground_truth_rows)}"
            ),
            flush=True,
        )

        print(
            (
                f"  Intermediate CSVs updated in:\n"
                f"    {OUTPUT_DIR}"
            ),
            flush=True,
        )

        if remaining_pairs > 0:
            print(
                (
                    f"  Approx. ETA: "
                    f"{eta_seconds / 60.0:.1f} min "
                    f"for {remaining_pairs} remaining pair(s)"
                ),
                flush=True,
            )

    # -----------------------------------------------------------------
    # Final object-level CSVs
    # -----------------------------------------------------------------

    save_csv(
        crosswave_rows,
        OUTPUT_DIR
        / "crosswave_object_features.csv",
    )

    save_csv(
        ground_truth_rows,
        OUTPUT_DIR
        / "ground_truth_object_features.csv",
    )

    save_csv(
        segmentation_summary,
        OUTPUT_DIR
        / "segmentation_summary_by_roi.csv",
    )

    print()
    print(
        "Step 3/4: comparing overall feature distributions...",
        flush=True,
    )

    overall = []

    for feature in FEATURES:
        result = compare_distributions(
            crosswave_rows,
            ground_truth_rows,
            feature,
            bins=
                args.hist_bins,
        )

        overall.append(
            result
        )

        print(
            (
                f"  {feature:30s} "
                f"overlap={result['overlap_coefficient']:.4f} "
                f"JS={result['js_divergence']:.4f} "
                f"KS={result['ks_statistic']:.4f} "
                f"d={result['cohens_d']:.4f}"
            ),
            flush=True,
        )

    save_csv(
        overall,
        OUTPUT_DIR
        / "distribution_metrics_overall.csv",
    )

    print()
    print(
        "Step 4/4: comparing distributions within each ROI...",
        flush=True,
    )

    per_roi = []

    for index, pair in enumerate(
        pairs,
        start=1,
    ):
        roi_id = pair[
            "roi_id"
        ]

        cw_roi = [
            row
            for row in crosswave_rows
            if row[
                "roi_id"
            ] == roi_id
        ]

        gt_roi = [
            row
            for row in ground_truth_rows
            if row[
                "roi_id"
            ] == roi_id
        ]

        for feature in FEATURES:
            result = compare_distributions(
                cw_roi,
                gt_roi,
                feature,
                bins=
                    args.hist_bins,
            )

            per_roi.append(
                {
                    "roi_id":
                        roi_id,
                    **result,
                }
            )

        print(
            (
                f"  Finished ROI statistics "
                f"{index}/{len(pairs)}: {roi_id}"
            ),
            flush=True,
        )

    save_csv(
        per_roi,
        OUTPUT_DIR
        / "distribution_metrics_by_roi.csv",
    )

    total_seconds = (
        time.time()
        - run_start
    )

    print()
    print(
        "=" * 78,
        flush=True,
    )

    print(
        "COMPLETE",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        (
            f"Total runtime: "
            f"{total_seconds / 60.0:.1f} min"
        ),
        flush=True,
    )

    print(
        (
            f"Final CrossWave objects: "
            f"{len(crosswave_rows)}"
        ),
        flush=True,
    )

    print(
        (
            f"Final ground-truth objects: "
            f"{len(ground_truth_rows)}"
        ),
        flush=True,
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
        flush=True,
    )


if __name__ == "__main__":
    main()

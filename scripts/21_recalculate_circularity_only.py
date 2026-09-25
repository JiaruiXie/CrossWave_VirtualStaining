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
# PROJECT PATHS
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

# IMPORTANT:
# These are kept identical to your current fast morphology script.
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

# Do NOT overwrite the existing morphology analysis.
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "morphology_crosswave_vs_gt_fast"
    / "circularity_corrected"
)


# ============================================================
# CSV HELPERS
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


# ============================================================
# READ MATCHED PAIRS
# ============================================================

def read_pairs():

    if not PAIR_MANIFEST.is_file():

        raise FileNotFoundError(
            f"Pair manifest not found:\n"
            f"{PAIR_MANIFEST}\n\n"
            "Run scripts/19_plot_subeval_pairs_separately.py first."
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
                f"CrossWave image not found:\n"
                f"{cw_path}"
            )

        if not gt_path.is_file():

            raise FileNotFoundError(
                f"Ground-truth image not found:\n"
                f"{gt_path}"
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
# H&E DECOMPOSITION
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
# SAME NUCLEUS SEGMENTATION AS FAST SCRIPT
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

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )
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

        return (
            np.zeros_like(
                filtered,
                dtype=np.int32,
            ),
            float(
                otsu
            ),
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

    return (
        output,
        float(
            otsu
        ),
    )


# ============================================================
# CONNECTED-COMPONENT CORRECTION
# ============================================================

def largest_connected_component(
    mask,
):
    """
    Keep only the largest 8-connected component belonging
    to one watershed object ID.

    This prevents area from being measured across multiple
    disconnected fragments while perimeter is measured from
    only one contour.
    """

    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    (
        count,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    # Background only
    if count <= 1:
        return None

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_id = (
        1
        + int(
            np.argmax(
                component_areas
            )
        )
    )

    component = (
        labels
        == largest_id
    ).astype(
        np.uint8
    )

    return component


# ============================================================
# CORRECTED CIRCULARITY ONLY
# ============================================================

def extract_corrected_circularity(
    labels,
    roi_id,
    domain,
):

    rows = []

    max_id = int(
        labels.max()
    )

    for object_id in range(
        1,
        max_id + 1,
    ):

        original_mask = (
            labels
            == object_id
        ).astype(
            np.uint8
        )

        original_pixel_area = int(
            original_mask.sum()
        )

        if original_pixel_area <= 0:
            continue

        # ----------------------------------------------------
        # FIX 1:
        # use ONE connected component only
        # ----------------------------------------------------

        component = (
            largest_connected_component(
                original_mask
            )
        )

        if component is None:
            continue

        component_pixel_area = int(
            component.sum()
        )

        if component_pixel_area <= 0:
            continue

        # ----------------------------------------------------
        # Extract contour from this SAME component
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        # ----------------------------------------------------
        # FIX 2:
        # area and perimeter from SAME contour geometry
        # ----------------------------------------------------

        contour_area = float(
            cv2.contourArea(
                contour
            )
        )

        contour_perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        # Invalid / degenerate contour
        if (
            contour_area <= 0
            or contour_perimeter <= 0
        ):

            corrected_circularity = (
                np.nan
            )

        else:

            corrected_circularity = (
                4.0
                * math.pi
                * contour_area
                / (
                    contour_perimeter
                    ** 2
                )
            )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        number_components, _, _, _ = (
            cv2.connectedComponentsWithStats(
                original_mask,
                connectivity=8,
            )
        )

        number_components = max(
            0,
            number_components - 1,
        )

        retained_fraction = (
            component_pixel_area
            / original_pixel_area
        )

        rows.append(
            {
                "domain":
                    domain,

                "roi_id":
                    roi_id,

                "object_id":
                    object_id,

                "nucleus_circularity":
                    corrected_circularity,

                # Diagnostic columns only:
                "original_pixel_area":
                    original_pixel_area,

                "largest_component_pixel_area":
                    component_pixel_area,

                "contour_area":
                    contour_area,

                "contour_perimeter":
                    contour_perimeter,

                "number_connected_components":
                    number_components,

                "largest_component_fraction":
                    retained_fraction,
            }
        )

    return rows


# ============================================================
# DISTRIBUTION COMPARISON — CIRCULARITY ONLY
# ============================================================

def circularity_values(
    rows,
):

    values = []

    for row in rows:

        try:
            value = float(
                row[
                    "nucleus_circularity"
                ]
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ):
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

    pooled_variance = (
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

    if pooled_variance <= 0:
        return 0.0

    return float(
        (
            x.mean()
            - y.mean()
        )
        / math.sqrt(
            pooled_variance
        )
    )


def compare_circularity(
    cw_rows,
    gt_rows,
    bins,
):

    x = circularity_values(
        cw_rows
    )

    y = circularity_values(
        gt_rows
    )

    result = {
        "feature":
            "nucleus_circularity",

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

        "crosswave_fraction_gt_1":
            np.nan,

        "ground_truth_fraction_gt_1":
            np.nan,

        "crosswave_max":
            np.nan,

        "ground_truth_max":
            np.nan,
    }

    if (
        x.size == 0
        or y.size == 0
    ):
        return result

    # --------------------------------------------------------
    # Same histogram procedure as original fast script
    # --------------------------------------------------------

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

    histogram_x, edges = np.histogram(
        x,
        bins=bins,
        range=(
            lo,
            hi,
        ),
    )

    histogram_y, _ = np.histogram(
        y,
        bins=edges,
    )

    p = histogram_x.astype(
        np.float64
    )

    q = histogram_y.astype(
        np.float64
    )

    if (
        p.sum() > 0
        and q.sum() > 0
    ):

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
        abs(
            d
        )
        if math.isfinite(
            d
        )
        else np.nan
    )

    result[
        "crosswave_fraction_gt_1"
    ] = float(
        np.mean(
            x > 1.0
        )
    )

    result[
        "ground_truth_fraction_gt_1"
    ] = float(
        np.mean(
            y > 1.0
        )
    )

    result[
        "crosswave_max"
    ] = float(
        np.max(
            x
        )
    )

    result[
        "ground_truth_max"
    ] = float(
        np.max(
            y
        )
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help=(
            "Must match your original fast morphology analysis. "
            "Default: 0.5."
        ),
    )

    parser.add_argument(
        "--min-nucleus-area",
        type=float,
        default=20.0,
        help=(
            "Minimum nucleus area in original-resolution pixels. "
            "Default: 20."
        ),
    )

    parser.add_argument(
        "--max-nucleus-area",
        type=float,
        default=2500.0,
        help=(
            "Maximum nucleus area in original-resolution pixels. "
            "Default: 2500."
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
        help=(
            "0 = all matched pairs."
        ),
    )

    args = parser.parse_args()

    if not (
        0.1
        <= args.scale
        <= 1.0
    ):

        raise ValueError(
            "--scale must be between 0.1 and 1.0."
        )

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
    print("CORRECTED NUCLEAR CIRCULARITY ONLY")
    print("=" * 78)

    print(
        f"Pairs: {len(pairs)}"
    )

    print(
        f"Scale: {args.scale}"
    )

    print(
        "No other morphology features will be recalculated."
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print()

    cw_rows = []
    gt_rows = []

    timing_rows = []

    run_start = time.time()

    # ========================================================
    # PROCESS ROIs
    # ========================================================

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

        timing_row = {
            "roi_id":
                pair[
                    "roi_id"
                ]
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

            # ------------------------------------------------
            # Load
            # ------------------------------------------------

            t0 = time.time()

            rgb = load_rgb(
                path,
                args.scale,
            )

            load_seconds = (
                time.time()
                - t0
            )

            # ------------------------------------------------
            # Segment nuclei — same as original fast script
            # ------------------------------------------------

            t0 = time.time()

            labels, otsu = segment_nuclei_fast(
                rgb,
                min_area_scaled,
                max_area_scaled,
                args.watershed_peak_fraction,
            )

            segment_seconds = (
                time.time()
                - t0
            )

            # ------------------------------------------------
            # Corrected circularity only
            # ------------------------------------------------

            t0 = time.time()

            rows = (
                extract_corrected_circularity(
                    labels=
                        labels,

                    roi_id=
                        pair[
                            "roi_id"
                        ],

                    domain=
                        domain,
                )
            )

            feature_seconds = (
                time.time()
                - t0
            )

            values = circularity_values(
                rows
            )

            fraction_gt_1 = (
                float(
                    np.mean(
                        values > 1.0
                    )
                )
                if values.size
                else np.nan
            )

            max_circularity = (
                float(
                    np.max(
                        values
                    )
                )
                if values.size
                else np.nan
            )

            multi_component_count = sum(
                int(
                    row[
                        "number_connected_components"
                    ] > 1
                )
                for row in rows
            )

            print(
                f"  {domain:13s} "
                f"n={len(rows):,} | "
                f"C>1={fraction_gt_1 * 100:.4f}% | "
                f"max C={max_circularity:.4f} | "
                f"multi-component={multi_component_count:,} | "
                f"segment={segment_seconds:.1f}s | "
                f"circularity={feature_seconds:.1f}s",
                flush=True,
            )

            timing_row[
                f"{domain}_objects"
            ] = len(
                rows
            )

            timing_row[
                f"{domain}_load_seconds"
            ] = load_seconds

            timing_row[
                f"{domain}_segmentation_seconds"
            ] = segment_seconds

            timing_row[
                f"{domain}_circularity_seconds"
            ] = feature_seconds

            if domain == "crosswave":

                cw_rows.extend(
                    rows
                )

            else:

                gt_rows.extend(
                    rows
                )

        timing_row[
            "pair_seconds"
        ] = (
            time.time()
            - pair_start
        )

        timing_rows.append(
            timing_row
        )

        # ----------------------------------------------------
        # Save partial results after each ROI
        # ----------------------------------------------------

        save_csv(
            cw_rows,
            OUTPUT_DIR
            / "crosswave_circularity_corrected_partial.csv",
        )

        save_csv(
            gt_rows,
            OUTPUT_DIR
            / "ground_truth_circularity_corrected_partial.csv",
        )

        save_csv(
            timing_rows,
            OUTPUT_DIR
            / "timing_partial.csv",
        )

        print(
            f"  DONE in "
            f"{timing_row['pair_seconds']:.1f}s",
            flush=True,
        )

        print()

    # ========================================================
    # SAVE FINAL OBJECT-LEVEL CIRCULARITY
    # ========================================================

    save_csv(
        cw_rows,
        OUTPUT_DIR
        / "crosswave_circularity_corrected.csv",
    )

    save_csv(
        gt_rows,
        OUTPUT_DIR
        / "ground_truth_circularity_corrected.csv",
    )

    save_csv(
        timing_rows,
        OUTPUT_DIR
        / "timing_by_roi.csv",
    )

    # ========================================================
    # OVERALL CIRCULARITY RESULT ONLY
    # ========================================================

    overall = compare_circularity(
        cw_rows,
        gt_rows,
        bins=
            args.hist_bins,
    )

    save_csv(
        [
            overall
        ],
        OUTPUT_DIR
        / "circularity_metrics_overall_corrected.csv",
    )

    # ========================================================
    # PER-ROI CIRCULARITY ONLY
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

        result = compare_circularity(
            cw_roi,
            gt_roi,
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

    save_csv(
        per_roi,
        OUTPUT_DIR
        / "circularity_metrics_by_roi_corrected.csv",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 78)
    print("CORRECTED CIRCULARITY RESULT")
    print("=" * 78)

    print(
        f"CrossWave n: "
        f"{overall['n_crosswave']:,}"
    )

    print(
        f"Ground truth n: "
        f"{overall['n_ground_truth']:,}"
    )

    print()

    print(
        f"Mean circularity:"
    )

    print(
        f"  CrossWave    = "
        f"{overall['mean_crosswave']:.6f}"
    )

    print(
        f"  Ground truth = "
        f"{overall['mean_ground_truth']:.6f}"
    )

    print()

    print(
        f"Overlap = "
        f"{overall['overlap_coefficient']:.4f}"
    )

    print(
        f"JS      = "
        f"{overall['js_divergence']:.4f}"
    )

    print(
        f"KS      = "
        f"{overall['ks_statistic']:.4f}"
    )

    print(
        f"p       = "
        f"{overall['ks_pvalue']:.3g}"
    )

    print(
        f"d       = "
        f"{overall['cohens_d']:.4f}"
    )

    print()

    print(
        "Fraction with circularity > 1:"
    )

    print(
        f"  CrossWave    = "
        f"{overall['crosswave_fraction_gt_1'] * 100:.6f}%"
    )

    print(
        f"  Ground truth = "
        f"{overall['ground_truth_fraction_gt_1'] * 100:.6f}%"
    )

    print()

    print(
        "Maximum corrected circularity:"
    )

    print(
        f"  CrossWave    = "
        f"{overall['crosswave_max']:.6f}"
    )

    print(
        f"  Ground truth = "
        f"{overall['ground_truth_max']:.6f}"
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
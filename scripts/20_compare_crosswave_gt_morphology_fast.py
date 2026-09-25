
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

PAIR_MANIFEST = ROOT / "outputs" / "comparison_figures" / "separate_pairs" / "generated_pair_comparisons.csv"
CROSSWAVE_DIR = ROOT / "outputs" / "crosswave_dapi_student_lr_v3" / "sub_eval_vhe"
GT_DIR = ROOT / "data" / "raw" / "Sub_eval" / "GroudTruth"
OUTPUT_DIR = ROOT / "outputs" / "morphology_crosswave_vs_gt_fast"

FEATURES = [
    "nucleus_max_caliper",
    "nucleus_perimeter",
    "nucleus_area",
    "nucleus_eccentricity",
    "nucleus_circularity",
    "nucleus_eosin_od_mean",
]


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_pairs():
    if not PAIR_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Pair manifest not found: {PAIR_MANIFEST}\n"
            "Run scripts/19_plot_subeval_pairs_separately.py first."
        )

    with PAIR_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    pairs = []
    for i, row in enumerate(rows, 1):
        cw_name = row.get("crosswave", "").strip()
        gt_name = row.get("ground_truth", "").strip()
        if not cw_name or not gt_name:
            continue

        cw = CROSSWAVE_DIR / cw_name
        gt = GT_DIR / gt_name

        if not cw.is_file():
            raise FileNotFoundError(cw)
        if not gt.is_file():
            raise FileNotFoundError(gt)

        pairs.append({"roi_id": f"roi_{i:03d}", "crosswave": cw, "ground_truth": gt})

    return pairs


def load_rgb(path, scale):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if scale != 1.0:
            new_size = (
                max(1, int(round(im.width * scale))),
                max(1, int(round(im.height * scale))),
            )
            im = im.resize(new_size, Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def he_channels(rgb):
    rgbf = np.clip(rgb.astype(np.float32) / 255.0, 1.0 / 255.0, 1.0)
    od = -np.log(rgbf)

    r = od[:, :, 0]
    g = od[:, :, 1]
    b = od[:, :, 2]

    h = 1.88000259 * r - 0.07359820 * g - 0.59521871 * b
    e = -1.01536655 * r + 1.13316777 * g - 0.48168478 * b

    return np.maximum(h, 0.0).astype(np.float32), np.maximum(e, 0.0).astype(np.float32)


def robust_uint8(image):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.uint8)

    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)

    x = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)


def segment_nuclei_fast(rgb, min_area, max_area, peak_fraction):
    h, e = he_channels(rgb)
    h8 = robust_uint8(h)
    h8 = cv2.GaussianBlur(h8, (3, 3), 0)

    otsu, binary = cv2.threshold(
        h8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    count, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )

    filtered = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[cc_labels == label] = 1

    dist = cv2.distanceTransform(filtered, cv2.DIST_L2, 5)

    if float(dist.max()) <= 0:
        return np.zeros_like(filtered, dtype=np.int32), e, float(otsu)

    local_max = dist == ndi.maximum_filter(dist, size=5, mode="nearest")
    local_max &= dist >= peak_fraction * float(dist.max())

    markers, n_markers = ndi.label(local_max)
    if n_markers == 0:
        markers, _ = ndi.label(filtered)

    markers = markers.astype(np.int32)
    ws_input = cv2.cvtColor(255 - h8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_input, markers)

    output = np.zeros_like(markers, dtype=np.int32)
    next_id = 1

    for old_id in np.unique(markers):
        if old_id <= 0:
            continue
        mask = (markers == old_id) & (filtered > 0)
        area = int(mask.sum())
        if min_area <= area <= max_area:
            output[mask] = next_id
            next_id += 1

    return output, e, float(otsu)


def eccentricity_from_contour(contour):
    if contour is None or len(contour) < 5:
        return np.nan
    try:
        ellipse = cv2.fitEllipse(contour)
        a = max(ellipse[1])
        b = min(ellipse[1])
        if a <= 0:
            return 0.0
        return math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
    except cv2.error:
        return np.nan


def extract_features_fast(labels, eosin, roi_id, domain, scale):
    rows = []

    area_factor = 1.0 / (scale * scale)
    length_factor = 1.0 / scale

    max_id = int(labels.max())
    for obj_id in range(1, max_id + 1):
        mask = (labels == obj_id).astype(np.uint8)
        area_scaled = int(mask.sum())
        if area_scaled <= 0:
            continue

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter_scaled = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * math.pi * area_scaled / (perimeter_scaled ** 2)
            if perimeter_scaled > 0
            else np.nan
        )

        rect = cv2.minAreaRect(contour)
        max_caliper = max(float(rect[1][0]), float(rect[1][1])) * length_factor

        e_vals = eosin[labels == obj_id]

        rows.append(
            {
                "domain": domain,
                "roi_id": roi_id,
                "object_id": obj_id,
                "nucleus_max_caliper": max_caliper,
                "nucleus_perimeter": perimeter_scaled * length_factor,
                "nucleus_area": area_scaled * area_factor,
                "nucleus_eccentricity": eccentricity_from_contour(contour),
                "nucleus_circularity": circularity,
                "nucleus_eosin_od_mean": float(e_vals.mean()) if e_vals.size else np.nan,
            }
        )

    return rows


def finite_values(rows, feature):
    vals = []
    for row in rows:
        try:
            v = float(row[feature])
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=np.float64)


def cohens_d(x, y):
    if x.size < 2 or y.size < 2:
        return np.nan

    pooled_var = (
        (x.size - 1) * x.var(ddof=1)
        + (y.size - 1) * y.var(ddof=1)
    ) / (x.size + y.size - 2)

    if pooled_var <= 0:
        return 0.0

    return float((x.mean() - y.mean()) / math.sqrt(float(pooled_var)))


def compare_distributions(cw_rows, gt_rows, feature, bins):
    x = finite_values(cw_rows, feature)
    y = finite_values(gt_rows, feature)

    out = {
        "feature": feature,
        "n_crosswave": int(x.size),
        "n_ground_truth": int(y.size),
        "mean_crosswave": float(x.mean()) if x.size else np.nan,
        "mean_ground_truth": float(y.mean()) if y.size else np.nan,
        "overlap_coefficient": np.nan,
        "js_divergence": np.nan,
        "ks_statistic": np.nan,
        "ks_pvalue": np.nan,
        "cohens_d": np.nan,
        "abs_cohens_d": np.nan,
    }

    if x.size == 0 or y.size == 0:
        return out

    combined = np.concatenate([x, y])
    lo = float(np.percentile(combined, 0.5))
    hi = float(np.percentile(combined, 99.5))
    if hi <= lo:
        hi = lo + 1e-6

    hx, edges = np.histogram(x, bins=bins, range=(lo, hi))
    hy, _ = np.histogram(y, bins=edges)

    p = hx.astype(np.float64)
    q = hy.astype(np.float64)

    if p.sum() > 0 and q.sum() > 0:
        p /= p.sum()
        q /= q.sum()

        out["overlap_coefficient"] = float(np.minimum(p, q).sum())

        eps = 1e-12
        p2 = np.clip(p, eps, None)
        q2 = np.clip(q, eps, None)
        p2 /= p2.sum()
        q2 /= q2.sum()
        m = 0.5 * (p2 + q2)

        out["js_divergence"] = float(
            0.5 * np.sum(p2 * np.log(p2 / m))
            + 0.5 * np.sum(q2 * np.log(q2 / m))
        )

    ks = ks_2samp(x, y, alternative="two-sided", method="auto")
    out["ks_statistic"] = float(ks.statistic)
    out["ks_pvalue"] = float(ks.pvalue)

    d = cohens_d(x, y)
    out["cohens_d"] = d
    out["abs_cohens_d"] = abs(d) if math.isfinite(d) else np.nan

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Analysis scale. 0.5 recommended; 1.0 = full resolution.",
    )
    parser.add_argument(
        "--min-nucleus-area",
        type=float,
        default=20.0,
        help="Minimum nucleus area in original-resolution pixels.",
    )
    parser.add_argument(
        "--max-nucleus-area",
        type=float,
        default=2500.0,
        help="Maximum nucleus area in original-resolution pixels.",
    )
    parser.add_argument(
        "--watershed-peak-fraction",
        type=float,
        default=0.12,
    )
    parser.add_argument("--hist-bins", type=int, default=50)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="0 = all pairs.",
    )

    args = parser.parse_args()

    if not 0.1 <= args.scale <= 1.0:
        raise ValueError("--scale must be between 0.1 and 1.0")

    min_area_scaled = max(
        3,
        int(round(args.min_nucleus_area * args.scale * args.scale)),
    )
    max_area_scaled = max(
        min_area_scaled + 1,
        int(round(args.max_nucleus_area * args.scale * args.scale)),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs()
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    print("=" * 78, flush=True)
    print("FAST CrossWave vs Ground Truth Morphology", flush=True)
    print("=" * 78, flush=True)
    print(f"Pairs: {len(pairs)}", flush=True)
    print(f"Analysis scale: {args.scale}", flush=True)
    print("Cell segmentation: disabled for speed", flush=True)
    print(
        f"Scaled nucleus area range: {min_area_scaled}..{max_area_scaled}",
        flush=True,
    )
    print()

    cw_rows = []
    gt_rows = []
    timing_rows = []

    pair_times = []
    run_start = time.time()

    for pair_index, pair in enumerate(pairs, 1):
        pair_start = time.time()

        print("-" * 78, flush=True)
        print(
            f"PAIR {pair_index}/{len(pairs)} | {pair['roi_id']}",
            flush=True,
        )

        timing_row = {"roi_id": pair["roi_id"]}

        for domain, path in (
            ("crosswave", pair["crosswave"]),
            ("ground_truth", pair["ground_truth"]),
        ):
            t0 = time.time()
            rgb = load_rgb(path, args.scale)
            load_s = time.time() - t0

            print(
                f"  {domain}: {rgb.shape[1]} x {rgb.shape[0]}",
                flush=True,
            )

            t0 = time.time()
            labels, eosin, otsu = segment_nuclei_fast(
                rgb,
                min_area_scaled,
                max_area_scaled,
                args.watershed_peak_fraction,
            )
            seg_s = time.time() - t0

            t0 = time.time()
            rows = extract_features_fast(
                labels,
                eosin,
                pair["roi_id"],
                domain,
                args.scale,
            )
            feat_s = time.time() - t0

            print(
                f"    nuclei={len(rows):,} | load={load_s:.1f}s | "
                f"segment={seg_s:.1f}s | features={feat_s:.1f}s | "
                f"Otsu={otsu:.1f}",
                flush=True,
            )

            if rows:
                mean_area = float(
                    np.mean([r["nucleus_area"] for r in rows])
                )
                mean_ecc = float(
                    np.nanmean([r["nucleus_eccentricity"] for r in rows])
                )
                print(
                    f"    check: mean area={mean_area:.2f}, "
                    f"mean eccentricity={mean_ecc:.3f}",
                    flush=True,
                )

            timing_row[f"{domain}_nuclei"] = len(rows)
            timing_row[f"{domain}_load_seconds"] = load_s
            timing_row[f"{domain}_segmentation_seconds"] = seg_s
            timing_row[f"{domain}_feature_seconds"] = feat_s

            if domain == "crosswave":
                cw_rows.extend(rows)
            else:
                gt_rows.extend(rows)

        pair_s = time.time() - pair_start
        timing_row["pair_seconds"] = pair_s
        timing_rows.append(timing_row)
        pair_times.append(pair_s)

        # Save partial object results after every ROI.
        save_csv(
            cw_rows,
            OUTPUT_DIR / "crosswave_object_features_partial.csv",
        )
        save_csv(
            gt_rows,
            OUTPUT_DIR / "ground_truth_object_features_partial.csv",
        )
        save_csv(
            timing_rows,
            OUTPUT_DIR / "timing_partial.csv",
        )

        # Save cumulative morphology statistics after every ROI.
        partial_metrics = [
            compare_distributions(
                cw_rows,
                gt_rows,
                feature,
                args.hist_bins,
            )
            for feature in FEATURES
        ]

        save_csv(
            partial_metrics,
            OUTPUT_DIR / "distribution_metrics_partial.csv",
        )

        remaining = len(pairs) - pair_index
        avg_pair_s = sum(pair_times) / len(pair_times)

        print(
            f"  DONE in {pair_s:.1f}s ({pair_s / 60.0:.2f} min)",
            flush=True,
        )
        print(
            f"  Running totals: CW={len(cw_rows):,}, GT={len(gt_rows):,}",
            flush=True,
        )

        if remaining:
            print(
                f"  ETA ≈ {avg_pair_s * remaining / 60.0:.1f} min",
                flush=True,
            )

        print(
            "  Partial metrics updated: distribution_metrics_partial.csv",
            flush=True,
        )
        print()

    save_csv(
        cw_rows,
        OUTPUT_DIR / "crosswave_object_features.csv",
    )
    save_csv(
        gt_rows,
        OUTPUT_DIR / "ground_truth_object_features.csv",
    )
    save_csv(
        timing_rows,
        OUTPUT_DIR / "timing_by_roi.csv",
    )

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
        OUTPUT_DIR / "distribution_metrics_overall.csv",
    )

    per_roi = []

    for pair in pairs:
        roi_id = pair["roi_id"]
        cw_roi = [r for r in cw_rows if r["roi_id"] == roi_id]
        gt_roi = [r for r in gt_rows if r["roi_id"] == roi_id]

        for feature in FEATURES:
            result = compare_distributions(
                cw_roi,
                gt_roi,
                feature,
                args.hist_bins,
            )
            per_roi.append({"roi_id": roi_id, **result})

    save_csv(
        per_roi,
        OUTPUT_DIR / "distribution_metrics_by_roi.csv",
    )

    print("=" * 78, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 78, flush=True)

    for row in overall:
        print(
            f"{row['feature']:28s} "
            f"overlap={row['overlap_coefficient']:.4f} "
            f"JS={row['js_divergence']:.4f} "
            f"KS={row['ks_statistic']:.4f} "
            f"p={row['ks_pvalue']:.3g} "
            f"d={row['cohens_d']:.4f}",
            flush=True,
        )

    print()
    print(
        f"Total runtime: {(time.time() - run_start) / 60.0:.2f} min",
        flush=True,
    )
    print(f"Saved to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

"""
Generate one separate side-by-side comparison PNG for each matched Sub_eval ROI.

Columns:
    MAF input | DAPI baseline | CrossWave | MAF teacher | Ground truth H&E

Each matched ROI is saved as its own PNG instead of combining multiple ROIs
into one large figure.

Optional metrics:
    --show-metrics true
    --show-metrics false

Examples
--------
Generate all matched pairs with metrics:

    & "C:/Users/jiaru/.conda/envs/crosswave_vs/python.exe" `
      scripts/19_plot_subeval_pairs_separately.py `
      --show-metrics true

Generate all matched pairs without metrics:

    & "C:/Users/jiaru/.conda/envs/crosswave_vs/python.exe" `
      scripts/19_plot_subeval_pairs_separately.py `
      --show-metrics false

Generate only the first 10 matched pairs:

    & "C:/Users/jiaru/.conda/envs/crosswave_vs/python.exe" `
      scripts/19_plot_subeval_pairs_separately.py `
      --max-samples 10

This script performs NO model inference.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


Image.MAX_IMAGE_PIXELS = None


# =====================================================================
# Project paths
# =====================================================================

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]

AF_DIR = ROOT / "data" / "raw" / "Sub_eval" / "AF"

BASELINE_DIR = (
    ROOT
    / "outputs"
    / "dapi_baseline_v1"
    / "sub_eval_vhe"
)

CROSSWAVE_DIR = (
    ROOT
    / "outputs"
    / "crosswave_dapi_student_newTeacher_v2"
    / "sub_eval_vhe"
)

TEACHER_DIR = (
    ROOT
    / "outputs"
    / "maf_teacher_stable_b_inverted_y_v2"
    / "sub_eval_vhe"
)

GROUND_TRUTH_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Sub_eval"
    / "GroudTruth"
)

BASELINE_EXPERIMENT = ROOT / "outputs" / "dapi_baseline_v1"
CROSSWAVE_EXPERIMENT = ROOT / "outputs" / "crosswave_dapi_student_newTeacher_v2"
TEACHER_EXPERIMENT = ROOT / "outputs" / "maf_teacher_stable_b_inverted_y_v2"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "comparison_figures"
    / "separate_pairs"
)

IMAGE_SUFFIXES = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}


# =====================================================================
# CLI helpers
# =====================================================================


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y", "on"}:
        return True

    if value in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError("Expected true/false.")


# =====================================================================
# Filename parsing / matching
# =====================================================================


MODEL_SUFFIX_PATTERNS = (
    r"_dapi_baseline_vhe$",
    r"_crosswave_v3_vhe$",
    r"_crosswave_vhe$",
    r"_teacher_vhe$",
    r"_virtual_he$",
    r"_virtual_h&e$",
    r"_vhe$",
)


def clean_stem_text(text):
    text = text.lower().strip()

    text = re.sub(
        r"\(\d+\)$",
        "",
        text,
    )

    for pattern in MODEL_SUFFIX_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalized_stem(path):
    return clean_stem_text(path.stem)


def canonical_semantic_tail(text):
    text = clean_stem_text(text)

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")

    if not text:
        return ""

    tokens = [
        token
        for token in text.split("_")
        if token
    ]

    modality_tokens = {
        "af",
        "maf",
        "he",
        "h&e",
        "vhe",
        "virtualhe",
        "virtual",
    }

    tokens = [
        token
        for token in tokens
        if token not in modality_tokens
    ]

    return "_".join(tokens)


def parse_roi(path):
    text = clean_stem_text(path.stem)

    sample_match = re.search(
        r"(sample\s*\d+)",
        text,
        flags=re.IGNORECASE,
    )

    sample = (
        sample_match.group(1)
        .replace(" ", "")
        .lower()
        if sample_match
        else ""
    )

    coord_match = re.search(
        r"x\s*=\s*(\d+).*?"
        r"y\s*=\s*(\d+).*?"
        r"w\s*=\s*(\d+).*?"
        r"h\s*=\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if coord_match is None:
        x = y = w = h = None
        exact_roi = None
        xy_roi = None
    else:
        x = int(coord_match.group(1))
        y = int(coord_match.group(2))
        w = int(coord_match.group(3))
        h = int(coord_match.group(4))

        exact_roi = (
            sample,
            x,
            y,
            w,
            h,
        )

        xy_roi = (
            sample,
            x,
            y,
        )

    tail = ""

    tail_match = re.search(
        r"\)\s*[_\-]\s*(.+)$",
        text,
    )

    if tail_match is not None:
        tail = tail_match.group(1)

    semantic_tail = canonical_semantic_tail(
        tail
    )

    semantic_key = (
        sample,
        semantic_tail,
    ) if semantic_tail else None

    return {
        "sample": sample,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "exact_roi": exact_roi,
        "xy_roi": xy_roi,
        "semantic_tail": semantic_tail,
        "semantic_key": semantic_key,
        "normalized_stem": normalized_stem(path),
    }


def list_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {folder}"
        )

    return sorted(
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    )


def add_unique(dictionary, key, value):
    if key is None:
        return

    if key not in dictionary:
        dictionary[key] = value
    else:
        dictionary[key] = None


def build_lookup(folder):
    images = list_images(folder)

    by_stem = {}
    by_exact_roi = {}
    by_xy_roi = {}
    by_semantic = {}
    parsed = []

    for path in images:
        info = parse_roi(path)

        parsed.append(
            {
                "path": path,
                "info": info,
            }
        )

        add_unique(
            by_stem,
            info["normalized_stem"],
            path,
        )

        add_unique(
            by_exact_roi,
            info["exact_roi"],
            path,
        )

        add_unique(
            by_xy_roi,
            info["xy_roi"],
            path,
        )

        add_unique(
            by_semantic,
            info["semantic_key"],
            path,
        )

    return {
        "images": images,
        "parsed": parsed,
        "by_stem": by_stem,
        "by_exact_roi": by_exact_roi,
        "by_xy_roi": by_xy_roi,
        "by_semantic": by_semantic,
    }


def bbox_iou(info_a, info_b):
    values_a = (
        info_a["x"],
        info_a["y"],
        info_a["w"],
        info_a["h"],
    )

    values_b = (
        info_b["x"],
        info_b["y"],
        info_b["w"],
        info_b["h"],
    )

    if (
        any(value is None for value in values_a)
        or any(value is None for value in values_b)
    ):
        return 0.0

    ax1 = info_a["x"]
    ay1 = info_a["y"]
    ax2 = ax1 + info_a["w"]
    ay2 = ay1 + info_a["h"]

    bx1 = info_b["x"]
    by1 = info_b["y"]
    bx2 = bx1 + info_b["w"]
    by2 = by1 + info_b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = info_a["w"] * info_a["h"]
    area_b = info_b["w"] * info_b["h"]

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def normalized_center_distance(info_a, info_b):
    values_a = (
        info_a["x"],
        info_a["y"],
        info_a["w"],
        info_a["h"],
    )

    values_b = (
        info_b["x"],
        info_b["y"],
        info_b["w"],
        info_b["h"],
    )

    if (
        any(value is None for value in values_a)
        or any(value is None for value in values_b)
    ):
        return float("inf")

    ax = info_a["x"] + 0.5 * info_a["w"]
    ay = info_a["y"] + 0.5 * info_a["h"]

    bx = info_b["x"] + 0.5 * info_b["w"]
    by = info_b["y"] + 0.5 * info_b["h"]

    distance = math.sqrt(
        (ax - bx) ** 2
        + (ay - by) ** 2
    )

    scale = max(
        1.0,
        0.5
        * (
            math.sqrt(
                info_a["w"] ** 2
                + info_a["h"] ** 2
            )
            + math.sqrt(
                info_b["w"] ** 2
                + info_b["h"] ** 2
            )
        ),
    )

    return distance / scale


def match_model_output(
    lookup,
    af_info,
):
    candidate = lookup["by_stem"].get(
        af_info["normalized_stem"]
    )

    if candidate is not None:
        return candidate, "normalized_stem"

    candidate = lookup["by_exact_roi"].get(
        af_info["exact_roi"]
    )

    if candidate is not None:
        return candidate, "sample+x+y+w+h"

    candidate = lookup["by_xy_roi"].get(
        af_info["xy_roi"]
    )

    if candidate is not None:
        return candidate, "sample+x+y"

    candidate = lookup["by_semantic"].get(
        af_info["semantic_key"]
    )

    if candidate is not None:
        return candidate, "sample+semantic_tail"

    return None, ""


def match_ground_truth(
    gt_lookup,
    af_info,
    max_normalized_center_distance=1.25,
):
    candidate = gt_lookup["by_stem"].get(
        af_info["normalized_stem"]
    )

    if candidate is not None:
        return candidate, "normalized_stem", ""

    candidate = gt_lookup["by_semantic"].get(
        af_info["semantic_key"]
    )

    if candidate is not None:
        return (
            candidate,
            "sample+canonical_semantic_tail",
            "",
        )

    candidate = gt_lookup["by_exact_roi"].get(
        af_info["exact_roi"]
    )

    if candidate is not None:
        return candidate, "sample+x+y+w+h", ""

    candidate = gt_lookup["by_xy_roi"].get(
        af_info["xy_roi"]
    )

    if candidate is not None:
        return candidate, "sample+x+y", ""

    same_sample = [
        item
        for item in gt_lookup["parsed"]
        if (
            item["info"]["sample"]
            == af_info["sample"]
        )
    ]

    if not same_sample:
        return (
            None,
            "",
            "no ground truth from same sample",
        )

    scored = []

    for item in same_sample:
        gt_info = item["info"]

        iou = bbox_iou(
            af_info,
            gt_info,
        )

        center_distance = normalized_center_distance(
            af_info,
            gt_info,
        )

        scored.append(
            (
                iou,
                center_distance,
                item["path"],
            )
        )

    scored.sort(
        key=lambda value: (
            -value[0],
            value[1],
        )
    )

    best_iou, best_distance, best_path = scored[0]

    if best_iou > 0:
        return (
            best_path,
            "same-sample-best-IoU",
            (
                f"IoU={best_iou:.4f}; "
                f"norm_center_distance={best_distance:.4f}"
            ),
        )

    if best_distance <= max_normalized_center_distance:
        return (
            best_path,
            "same-sample-nearest-center",
            (
                f"IoU=0; "
                f"norm_center_distance={best_distance:.4f}"
            ),
        )

    return (
        None,
        "",
        (
            "nearest same-sample ROI too far: "
            f"norm_center_distance={best_distance:.4f}"
        ),
    )


def match_five_way(
    af_dir,
    baseline_dir,
    crosswave_dir,
    teacher_dir,
    ground_truth_dir,
    max_normalized_center_distance,
):
    af_images = list_images(
        af_dir
    )

    baseline_lookup = build_lookup(
        baseline_dir
    )

    crosswave_lookup = build_lookup(
        crosswave_dir
    )

    teacher_lookup = build_lookup(
        teacher_dir
    )

    gt_lookup = build_lookup(
        ground_truth_dir
    )

    matches = []
    report = []

    used_gt_paths = set()

    for af_path in af_images:
        af_info = parse_roi(
            af_path
        )

        baseline_path, baseline_mode = match_model_output(
            baseline_lookup,
            af_info,
        )

        crosswave_path, crosswave_mode = match_model_output(
            crosswave_lookup,
            af_info,
        )

        teacher_path, teacher_mode = match_model_output(
            teacher_lookup,
            af_info,
        )

        gt_path, gt_mode, gt_note = match_ground_truth(
            gt_lookup,
            af_info,
            max_normalized_center_distance=
                max_normalized_center_distance,
        )

        duplicate_gt = (
            gt_path is not None
            and gt_path in used_gt_paths
        )

        complete = (
            baseline_path is not None
            and crosswave_path is not None
            and teacher_path is not None
            and gt_path is not None
            and not duplicate_gt
        )

        status = (
            "matched"
            if complete
            else "unmatched"
        )

        if duplicate_gt:
            status = (
                "unmatched_duplicate_ground_truth"
            )

        report.append(
            {
                "status":
                    status,
                "af":
                    af_path.name,
                "af_semantic":
                    af_info["semantic_tail"],
                "baseline":
                    baseline_path.name
                    if baseline_path
                    else "",
                "baseline_match":
                    baseline_mode,
                "crosswave":
                    crosswave_path.name
                    if crosswave_path
                    else "",
                "crosswave_match":
                    crosswave_mode,
                "teacher":
                    teacher_path.name
                    if teacher_path
                    else "",
                "teacher_match":
                    teacher_mode,
                "ground_truth":
                    gt_path.name
                    if gt_path
                    else "",
                "ground_truth_match":
                    gt_mode,
                "ground_truth_note":
                    gt_note,
            }
        )

        if complete:
            used_gt_paths.add(
                gt_path
            )

            matches.append(
                {
                    "info":
                        af_info,
                    "af":
                        af_path,
                    "baseline":
                        baseline_path,
                    "crosswave":
                        crosswave_path,
                    "teacher":
                        teacher_path,
                    "ground_truth":
                        gt_path,
                }
            )

    return matches, report


# =====================================================================
# Metric extraction
# =====================================================================


def find_history_csv(experiment_dir):
    candidates = list(
        experiment_dir.rglob(
            "training_history*.csv"
        )
    )

    if not candidates:
        candidates = list(
            experiment_dir.rglob(
                "*history*.csv"
            )
        )

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path:
            path.stat().st_mtime,
    )


def numeric(row, names):
    for name in names:
        raw = row.get(name)

        if raw is None:
            continue

        raw = str(raw).strip()

        if not raw:
            continue

        try:
            value = float(raw)
        except ValueError:
            continue

        if math.isfinite(value):
            return value

    return None


def read_best_metrics(experiment_dir):
    history_path = find_history_csv(
        experiment_dir
    )

    if history_path is None:
        return {
            "history": None,
            "epoch": None,
            "fid": None,
            "kid_mean": None,
            "kid_std": None,
        }

    with history_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    valid = []

    for row in rows:
        fid = numeric(
            row,
            (
                "sub_eval_fid",
                "fid",
                "FID",
            ),
        )

        if fid is not None:
            valid.append(
                (
                    fid,
                    row,
                )
            )

    if not valid:
        return {
            "history": history_path,
            "epoch": None,
            "fid": None,
            "kid_mean": None,
            "kid_std": None,
        }

    best_fid, best_row = min(
        valid,
        key=lambda item:
            item[0],
    )

    epoch = numeric(
        best_row,
        (
            "epoch",
            "Epoch",
        ),
    )

    kid_mean = numeric(
        best_row,
        (
            "sub_eval_kid_mean",
            "kid_mean",
            "KID_mean",
            "kid",
            "KID",
        ),
    )

    kid_std = numeric(
        best_row,
        (
            "sub_eval_kid_std",
            "kid_std",
            "KID_std",
        ),
    )

    return {
        "history":
            history_path,
        "epoch":
            int(epoch)
            if epoch is not None
            else None,
        "fid":
            best_fid,
        "kid_mean":
            kid_mean,
        "kid_std":
            kid_std,
    }


def metric_lines(
    name,
    metrics,
    show_metrics,
):
    if not show_metrics:
        return [name]

    lines = [
        name
    ]

    fid = metrics["fid"]
    kid_mean = metrics["kid_mean"]
    kid_std = metrics["kid_std"]

    if fid is None:
        lines.append(
            "FID/KID unavailable"
        )

        return lines

    lines.append(
        f"FID = {fid:.2f}"
    )

    if kid_mean is not None:
        if kid_std is not None:
            lines.append(
                (
                    f"KID = {kid_mean:.5f} "
                    f"± {kid_std:.5f}"
                )
            )
        else:
            lines.append(
                f"KID = {kid_mean:.5f}"
            )

    return lines


# =====================================================================
# Pillow rendering
# =====================================================================


def get_font(
    size,
    bold=False,
):
    if bold:
        candidates = [
            Path(
                "C:/Windows/Fonts/arialbd.ttf"
            ),
            Path(
                "C:/Windows/Fonts/calibrib.ttf"
            ),
        ]
    else:
        candidates = [
            Path(
                "C:/Windows/Fonts/arial.ttf"
            ),
            Path(
                "C:/Windows/Fonts/calibri.ttf"
            ),
        ]

    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(
                    str(path),
                    size=size,
                )
            except OSError:
                pass

    return ImageFont.load_default()


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert(
            "RGB"
        ).copy()


def fit_into_cell(
    image,
    cell_width,
    cell_height,
):
    fitted = ImageOps.contain(
        image,
        (
            cell_width,
            cell_height,
        ),
        method=
            Image.Resampling.LANCZOS,
    )

    cell = Image.new(
        "RGB",
        (
            cell_width,
            cell_height,
        ),
        "white",
    )

    x = (
        cell_width
        - fitted.width
    ) // 2

    y = (
        cell_height
        - fitted.height
    ) // 2

    cell.paste(
        fitted,
        (
            x,
            y,
        ),
    )

    return cell


def centered_multiline_text(
    draw,
    box,
    lines,
    title_font,
    metric_font,
):
    left, top, right, bottom = box

    line_spacing = 6
    heights = []

    for index, line in enumerate(
        lines
    ):
        font = (
            title_font
            if index == 0
            else metric_font
        )

        bbox = draw.textbbox(
            (0, 0),
            line,
            font=font,
        )

        heights.append(
            bbox[3] - bbox[1]
        )

    total_height = (
        sum(heights)
        + line_spacing
        * (
            len(lines)
            - 1
        )
    )

    current_y = (
        top
        + (
            bottom
            - top
            - total_height
        ) // 2
    )

    for index, line in enumerate(
        lines
    ):
        font = (
            title_font
            if index == 0
            else metric_font
        )

        bbox = draw.textbbox(
            (0, 0),
            line,
            font=font,
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        x = (
            left
            + (
                right
                - left
                - text_width
            ) // 2
        )

        draw.text(
            (
                x,
                current_y,
            ),
            line,
            fill="black",
            font=font,
        )

        current_y += (
            heights[index]
            + line_spacing
        )


def safe_filename(
    text,
):
    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip(
        "_."
    )


def output_name_for_pair(
    index,
    item,
):
    info = item["info"]

    sample = (
        info["sample"]
        or f"roi{index:03d}"
    )

    semantic = (
        info["semantic_tail"]
        or "roi"
    )

    parts = [
        f"{index:03d}",
        sample,
        semantic,
    ]

    if (
        info["x"] is not None
        and info["y"] is not None
    ):
        parts.extend(
            [
                f"x{info['x']}",
                f"y{info['y']}",
            ]
        )

    return (
        safe_filename(
            "_".join(parts)
        )
        + ".png"
    )


def build_single_pair_figure(
    item,
    baseline_metrics,
    crosswave_metrics,
    teacher_metrics,
    show_metrics,
    output_path,
    cell_width,
    cell_height,
):
    columns = [
        (
            "af",
            [
                "MAF input"
            ],
        ),
        (
            "baseline",
            metric_lines(
                "DAPI baseline",
                baseline_metrics,
                show_metrics,
            ),
        ),
        (
            "crosswave",
            metric_lines(
                "CrossWave",
                crosswave_metrics,
                show_metrics,
            ),
        ),
        (
            "teacher",
            metric_lines(
                "MAF teacher",
                teacher_metrics,
                show_metrics,
            ),
        ),
        (
            "ground_truth",
            [
                "Ground truth H&E"
            ],
        ),
    ]

    header_height = (
        125
        if show_metrics
        else 75
    )

    info_height = 54
    column_gap = 10
    outer_margin = 20

    number_columns = len(
        columns
    )

    canvas_width = (
        outer_margin * 2
        + number_columns * cell_width
        + (number_columns - 1) * column_gap
    )

    canvas_height = (
        outer_margin * 2
        + header_height
        + cell_height
        + info_height
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    title_font = get_font(
        26,
        bold=True,
    )

    metric_font = get_font(
        18,
        bold=False,
    )

    info_font = get_font(
        18,
        bold=False,
    )

    # -------------------------------------------------------------
    # Column headers
    # -------------------------------------------------------------

    for column_index, (
        _,
        lines,
    ) in enumerate(
        columns
    ):
        x0 = (
            outer_margin
            + column_index
            * (
                cell_width
                + column_gap
            )
        )

        x1 = x0 + cell_width

        centered_multiline_text(
            draw,
            (
                x0,
                outer_margin,
                x1,
                outer_margin + header_height,
            ),
            lines,
            title_font,
            metric_font,
        )

    # -------------------------------------------------------------
    # Images
    # -------------------------------------------------------------

    image_top = (
        outer_margin
        + header_height
    )

    for column_index, (
        key,
        _,
    ) in enumerate(
        columns
    ):
        x = (
            outer_margin
            + column_index
            * (
                cell_width
                + column_gap
            )
        )

        image = load_rgb(
            item[key]
        )

        cell = fit_into_cell(
            image,
            cell_width,
            cell_height,
        )

        canvas.paste(
            cell,
            (
                x,
                image_top,
            ),
        )

    # -------------------------------------------------------------
    # ROI identifier at bottom
    # -------------------------------------------------------------

    info = item["info"]

    label_parts = []

    if info["sample"]:
        label_parts.append(
            info["sample"]
        )

    if info["semantic_tail"]:
        label_parts.append(
            info["semantic_tail"]
        )

    if (
        info["x"] is not None
        and info["y"] is not None
    ):
        label_parts.append(
            (
                f"x={info['x']}, "
                f"y={info['y']}, "
                f"w={info['w']}, "
                f"h={info['h']}"
            )
        )

    label = (
        " | ".join(
            label_parts
        )
        if label_parts
        else output_path.stem
    )

    bbox = draw.textbbox(
        (0, 0),
        label,
        font=info_font,
    )

    text_width = (
        bbox[2] - bbox[0]
    )

    bottom_y = (
        image_top
        + cell_height
        + (
            info_height
            - (
                bbox[3] - bbox[1]
            )
        ) // 2
    )

    draw.text(
        (
            max(
                outer_margin,
                (
                    canvas_width
                    - text_width
                ) // 2,
            ),
            bottom_y,
        ),
        label,
        fill="black",
        font=info_font,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        format="PNG",
        optimize=False,
    )


# =====================================================================
# Reports
# =====================================================================


def save_csv(
    rows,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Maximum number of matched pairs to save. "
            "Use 0 to save all matched pairs. Default: 0."
        ),
    )

    parser.add_argument(
        "--show-metrics",
        type=str_to_bool,
        default=True,
        help=(
            "Display FID/KID in model headers: true or false."
        ),
    )

    parser.add_argument(
        "--cell-width",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--cell-height",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--max-gt-center-distance",
        type=float,
        default=1.25,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing pair-comparison PNGs."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Step 1/4: matching AF-anchored five-way ROI sets...",
        flush=True,
    )

    matches, report = match_five_way(
        af_dir=
            AF_DIR,
        baseline_dir=
            BASELINE_DIR,
        crosswave_dir=
            CROSSWAVE_DIR,
        teacher_dir=
            TEACHER_DIR,
        ground_truth_dir=
            GROUND_TRUTH_DIR,
        max_normalized_center_distance=
            args.max_gt_center_distance,
    )

    match_report_path = (
        OUTPUT_DIR
        / "matching_report.csv"
    )

    save_csv(
        report,
        match_report_path,
    )

    print(
        "Matched pairs:",
        len(matches),
        flush=True,
    )

    print(
        "Matching report:",
        match_report_path,
        flush=True,
    )

    if not matches:
        raise RuntimeError(
            "No five-way matched pairs were found."
        )

    print(
        "Step 2/4: reading FID/KID...",
        flush=True,
    )

    baseline_metrics = read_best_metrics(
        BASELINE_EXPERIMENT
    )

    crosswave_metrics = read_best_metrics(
        CROSSWAVE_EXPERIMENT
    )

    teacher_metrics = read_best_metrics(
        TEACHER_EXPERIMENT
    )

    print(
        "Show metrics:",
        args.show_metrics,
        flush=True,
    )

    if (
        args.max_samples > 0
        and len(matches) > args.max_samples
    ):
        selected = matches[
            :args.max_samples
        ]
    else:
        selected = matches

    print(
        f"Step 3/4: saving {len(selected)} separate comparison PNG(s)...",
        flush=True,
    )

    generated_report = []

    for index, item in enumerate(
        selected,
        start=1,
    ):
        filename = output_name_for_pair(
            index,
            item,
        )

        output_path = (
            OUTPUT_DIR
            / filename
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            print(
                f"[{index:03d}/{len(selected):03d}] "
                f"SKIP existing: {filename}",
                flush=True,
            )

            status = "skipped_existing"
        else:
            print(
                f"[{index:03d}/{len(selected):03d}] "
                f"Saving: {filename}",
                flush=True,
            )

            build_single_pair_figure(
                item=
                    item,
                baseline_metrics=
                    baseline_metrics,
                crosswave_metrics=
                    crosswave_metrics,
                teacher_metrics=
                    teacher_metrics,
                show_metrics=
                    args.show_metrics,
                output_path=
                    output_path,
                cell_width=
                    args.cell_width,
                cell_height=
                    args.cell_height,
            )

            if not output_path.is_file():
                raise RuntimeError(
                    f"Failed to create: {output_path}"
                )

            status = "saved"

        generated_report.append(
            {
                "index":
                    index,
                "status":
                    status,
                "comparison_png":
                    filename,
                "af":
                    item["af"].name,
                "baseline":
                    item["baseline"].name,
                "crosswave":
                    item["crosswave"].name,
                "teacher":
                    item["teacher"].name,
                "ground_truth":
                    item["ground_truth"].name,
            }
        )

    print(
        "Step 4/4: writing generated-pair manifest...",
        flush=True,
    )

    generated_report_path = (
        OUTPUT_DIR
        / "generated_pair_comparisons.csv"
    )

    save_csv(
        generated_report,
        generated_report_path,
    )

    print()
    print(
        "Complete.",
        flush=True,
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
        flush=True,
    )

    print(
        "Comparison images:",
        len(
            generated_report
        ),
        flush=True,
    )

    print(
        "Manifest:",
        generated_report_path,
        flush=True,
    )


if __name__ == "__main__":
    main()

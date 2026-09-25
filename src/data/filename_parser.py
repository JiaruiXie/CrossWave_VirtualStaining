import re
from pathlib import Path


def parse_training_maf_filename(path):
    """
    Parse a 256x256 Breast_Training MAF patch filename.

    Example
    -------
    sample1 (1, x=24916, y=33063, w=1821, h=1800)_
    AISD_20X_AF_1.tif_y00448_x00672.tif
    """

    name = Path(path).name

    sample_match = re.search(
        r"sample\s*(\d+)",
        name,
        re.IGNORECASE,
    )

    roi_match = re.search(
        r"\(\s*\d+\s*,\s*"
        r"x=(\d+)\s*,\s*"
        r"y=(\d+)\s*,\s*"
        r"w=(\d+)\s*,\s*"
        r"h=(\d+)\s*\)",
        name,
    )

    patch_match = re.search(
        r"_y(\d+)_x(\d+)\.tif$",
        name,
        re.IGNORECASE,
    )

    if not (
        sample_match
        and roi_match
        and patch_match
    ):
        return None

    sample = int(sample_match.group(1))

    roi_x = int(roi_match.group(1))
    roi_y = int(roi_match.group(2))
    roi_w = int(roi_match.group(3))
    roi_h = int(roi_match.group(4))

    patch_y = int(patch_match.group(1))
    patch_x = int(patch_match.group(2))

    return {
        "sample": sample,

        "roi_x": roi_x,
        "roi_y": roi_y,
        "roi_w": roi_w,
        "roi_h": roi_h,

        "patch_x": patch_x,
        "patch_y": patch_y,

        # Coordinates relative to scanned-image bounds
        "global_x": roi_x + patch_x,
        "global_y": roi_y + patch_y,
    }


def parse_subeval_maf_filename(path):
    """
    Parse a large Sub_eval MAF ROI.

    Example
    -------
    sample1 (1, x=24916, y=33063,
             w=1821, h=1800)_AISD_20X_AF_1.tif
    """

    name = Path(path).name

    sample_match = re.search(
        r"sample\s*(\d+)",
        name,
        re.IGNORECASE,
    )

    roi_match = re.search(
        r"\(\s*\d+\s*,\s*"
        r"x=(\d+)\s*,\s*"
        r"y=(\d+)\s*,\s*"
        r"w=(\d+)\s*,\s*"
        r"h=(\d+)\s*\)",
        name,
    )

    if not (
        sample_match
        and roi_match
    ):
        return None

    return {
        "sample": int(sample_match.group(1)),

        "roi_x": int(roi_match.group(1)),
        "roi_y": int(roi_match.group(2)),
        "roi_w": int(roi_match.group(3)),
        "roi_h": int(roi_match.group(4)),
    }


# Mapping empirically validated against raw MRXS
# using MAF-blue ↔ MRXS-DAPI NCC.
BREAST_SAMPLE_MAP = {
    1: 1,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
}
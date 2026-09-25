from pathlib import Path

import numpy as np
import openslide


def get_slide_bounds(slide):
    """
    Return OpenSlide tissue-bound offset.
    """

    bx = int(
        slide.properties.get(
            "openslide.bounds-x",
            0,
        )
    )

    by = int(
        slide.properties.get(
            "openslide.bounds-y",
            0,
        )
    )

    return bx, by


def extract_dapi_patch(
    slide,
    relative_x,
    relative_y,
    width=256,
    height=256,
):
    """
    Extract DAPI-filter autofluorescence from MRXS.

    Coordinates supplied here are relative to
    OpenSlide's scanned tissue bounds.
    """

    bx, by = get_slide_bounds(slide)

    x = bx + relative_x
    y = by + relative_y

    rgb = np.asarray(
        slide.read_region(& "C:\Users\jiaru\.conda\envs\crosswave_vs\python.exe" -X faulthandler `
  ".\audit_register_subeval_cropped.py" `
  --pairs all `
  --reuse-transforms `
  --force
            (x, y),
            level=0,
            size=(width, height),
        ).convert("RGB")
    )

    # Empirically validated:
    # MRXS blue plane corresponds to released
    # MAF DAPI/blue component.
    dapi = rgb[..., 2]

    return dapi, x, y


def open_breast_slide(
    raw_af_dir,
    raw_sample,
):
    path = (
        Path(raw_af_dir)
        / f"sample_{raw_sample:02d}.mrxs"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return openslide.OpenSlide(str(path))
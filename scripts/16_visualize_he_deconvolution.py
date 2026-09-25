import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

def to_uint8_rgb(image):
    """
    HWC float image [0,1] -> uint8 RGB.
    """

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return (
        image * 255.0
    ).astype(
        np.uint8
    )


def gray_to_rgb_uint8(image):
    """
    Float grayscale image [0,1]
    -> uint8 RGB.
    """

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = (
        image * 255.0
    ).astype(
        np.uint8
    )

    return np.stack(
        [
            image,
            image,
            image,
        ],
        axis=-1,
    )


def make_labeled_panel(
    image,
    label,
):
    """
    Create a 256x256 image with a small label strip.
    """

    if image.ndim == 2:
        image = gray_to_rgb_uint8(
            image
        )
    else:
        image = to_uint8_rgb(
            image
        )

    pil_image = Image.fromarray(
        image,
        mode="RGB",
    )

    label_height = 28

    panel = Image.new(
        "RGB",
        (
            pil_image.width,
            pil_image.height
            + label_height,
        ),
        "white",
    )

    panel.paste(
        pil_image,
        (
            0,
            label_height,
        ),
    )

    draw = ImageDraw.Draw(
        panel
    )

    draw.text(
        (
            6,
            7,
        ),
        label,
        fill="black",
    )

    return panel


def make_diagnostic_montage(
    maf_preview,
    c0,
    c2,
    fake_rgb,
    hematoxylin_display,
    eosin_display,
    correlations,
):
    """
    2x3 diagnostic montage without matplotlib.
    """

    corr_c0_h = correlations[
        "c0_h"
    ]

    corr_c0_e = correlations[
        "c0_e"
    ]

    corr_c2_h = correlations[
        "c2_h"
    ]

    corr_c2_e = correlations[
        "c2_e"
    ]

    panels = [
        make_labeled_panel(
            maf_preview,
            "MAF preview",
        ),

        make_labeled_panel(
            c0,
            f"C0 | H r={corr_c0_h:+.3f}",
        ),

        make_labeled_panel(
            c2,
            f"C2 | H r={corr_c2_h:+.3f}",
        ),

        make_labeled_panel(
            fake_rgb,
            "Generated H&E",
        ),

        make_labeled_panel(
            hematoxylin_display,
            "Hematoxylin OD",
        ),

        make_labeled_panel(
            eosin_display,
            "Eosin OD",
        ),
    ]

    panel_width = panels[
        0
    ].width

    panel_height = panels[
        0
    ].height

    footer_height = 38

    montage = Image.new(
        "RGB",
        (
            panel_width * 3,
            panel_height * 2
            + footer_height,
        ),
        "white",
    )

    for i, panel in enumerate(
        panels
    ):

        row = i // 3
        col = i % 3

        montage.paste(
            panel,
            (
                col * panel_width,
                row * panel_height,
            ),
        )

    draw = ImageDraw.Draw(
        montage
    )

    footer = (
        f"C0-H {corr_c0_h:+.3f} | "
        f"C0-E {corr_c0_e:+.3f} | "
        f"C2-H {corr_c2_h:+.3f} | "
        f"C2-E {corr_c2_e:+.3f}"
    )

    draw.text(
        (
            8,
            panel_height * 2
            + 10,
        ),
        footer,
        fill="black",
    )

    return montage

# =========================================================
# Project setup
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.data.datasets import (
    SingleDomainDataset,
)

from src.models.ecyclegan import (
    ECycleGAN,
)


# =========================================================
# Configuration
# =========================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "maf_teacher_stable_b"
    / "checkpoints"
    / "best_fid.pt"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "data_split.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maf_teacher_stable_b"
    / "he_deconvolution"
)

NUM_SAMPLES = 8


# =========================================================
# Tensor utilities
# =========================================================

def tensor_to_01(
    tensor,
):
    """
    Convert CHW tensor from [-1,1] to [0,1].
    """

    x = tensor.detach().cpu().float()

    x = (
        x + 1.0
    ) / 2.0

    x = torch.clamp(
        x,
        0.0,
        1.0,
    )

    return x.numpy()


def maf_to_preview(
    maf,
):
    """
    MAF is [2,H,W]:

        channel 0 = original C0
        channel 1 = original C2

    Preview reconstruction:

        R = C0
        G = 0
        B = C2
    """

    maf = tensor_to_01(
        maf
    )

    c0 = maf[0]
    c2 = maf[1]

    preview = np.zeros(
        (
            maf.shape[1],
            maf.shape[2],
            3,
        ),
        dtype=np.float32,
    )

    preview[
        :,
        :,
        0,
    ] = c0

    preview[
        :,
        :,
        2,
    ] = c2

    return preview


def fake_y_to_rgb(
    fake_y,
):
    """
    Convert generated H&E tensor from CHW [-1,1]
    to HWC RGB [0,1].
    """

    image = tensor_to_01(
        fake_y
    )

    image = np.transpose(
        image,
        (
            1,
            2,
            0,
        ),
    )

    return np.clip(
        image,
        0.0,
        1.0,
    )


# =========================================================
# H&E deconvolution
# =========================================================

def deconvolve_he(rgb):
    """
    H&E color deconvolution using explicit optical-density equations.

    Avoids:
        - skimage
        - np.linalg
        - matrix multiplication / BLAS

    rgb:
        H x W x 3 float image in [0, 1]

    Returns:
        hematoxylin concentration map
        eosin concentration map

    Higher values mean stronger stain.
    """

    rgb = np.asarray(
        rgb,
        dtype=np.float32,
    )

    rgb = np.clip(
        rgb,
        1.0 / 255.0,
        1.0,
    )

    # RGB -> optical density
    od = -np.log(
        rgb
    )

    od_r = od[:, :, 0]
    od_g = od[:, :, 1]
    od_b = od[:, :, 2]

    # Explicit transform using the same
    # precomputed inverse stain matrix.

    hematoxylin = (
        1.88000259 * od_r
        - 0.07359820 * od_g
        - 0.59521871 * od_b
    )

    eosin = (
        -1.01536655 * od_r
        + 1.13316777 * od_g
        - 0.48168478 * od_b
    )

    return (
        hematoxylin.astype(np.float32),
        eosin.astype(np.float32),
    )


def normalize_for_display(
    x,
    low_percentile=1,
    high_percentile=99,
):
    """
    Robust display normalization only.

    Do NOT use this normalized image for quantitative
    correlation calculations later.
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    low = np.percentile(
        x,
        low_percentile,
    )

    high = np.percentile(
        x,
        high_percentile,
    )

    if high <= low:

        return np.zeros_like(
            x,
            dtype=np.float32,
        )

    x = (
        x - low
    ) / (
        high - low
    )

    return np.clip(
        x,
        0.0,
        1.0,
    )


# =========================================================
# Load checkpoint
# =========================================================

def load_model(
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch"
        ),
    )

    print(
        "Checkpoint best FID:",
        checkpoint.get(
            "best_fid"
        ),
    )

    config = checkpoint[
        "config"
    ]

    model_cfg = config[
        "model"
    ]

    model = ECycleGAN(

        x_channels=
            model_cfg[
                "x_channels"
            ],

        y_channels=
            model_cfg[
                "y_channels"
            ],

        base_channels=
            model_cfg.get(
                "base_channels",
                64,
            ),

        num_res_blocks=
            model_cfg.get(
                "num_res_blocks",
                9,
            ),

        padding_type=
            model_cfg.get(
                "padding_type",
                "reflect",
            ),

    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        config,
    )


def pearson_corr(a, b):
    """
    Pearson correlation without np.corrcoef / np.cov.

    Uses only elementwise NumPy operations to avoid
    the crashing linear-algebra backend in this environment.
    """

    a = np.asarray(
        a,
        dtype=np.float32,
    ).ravel()

    b = np.asarray(
        b,
        dtype=np.float32,
    ).ravel()

    if a.size != b.size:
        raise ValueError(
            f"Size mismatch: {a.size} vs {b.size}"
        )

    a_mean = float(
        np.mean(a)
    )

    b_mean = float(
        np.mean(b)
    )

    da = a - a_mean
    db = b - b_mean

    numerator = float(
        np.sum(
            da * db
        )
    )

    denominator = float(
        np.sqrt(
            np.sum(
                da * da
            )
            *
            np.sum(
                db * db
            )
        )
    )

    if denominator <= 1e-12:
        return np.nan

    return (
        numerator
        / denominator
    )

# =========================================================
# Main
# =========================================================

@torch.no_grad()
def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model, config = load_model(
        CHECKPOINT,
        device,
    )

    # -----------------------------------------------------
    # Validation MAF dataset
    #
    # Deterministic: no augmentation.
    # -----------------------------------------------------

    dataset = SingleDomainDataset(
        split_csv=
            SPLIT_CSV,

        split="val",

        domain="x",

        mode="maf",
    )

    print(
        "Validation MAF images:",
        len(
            dataset
        ),
    )

    # -----------------------------------------------------
    # Use evenly spaced samples rather than only first N.
    # -----------------------------------------------------

    indices = np.linspace(
        0,
        len(dataset) - 1,
        NUM_SAMPLES,
        dtype=int,
    )

    for sample_number, index in enumerate(
        indices
    ):

        item = dataset[
            int(
                index
            )
        ]

        real_x = item[
            "image"
        ]

        model_x = real_x.unsqueeze(
            0
        ).to(
            device
        )

        # -------------------------------------------------
        # MAF -> virtual H&E
        # -------------------------------------------------

        fake_y = model.G_XY(
            model_x
        )[0]

        # -------------------------------------------------
        # Convert for analysis
        # -------------------------------------------------

        maf_01 = tensor_to_01(
            real_x
        )

        c0 = maf_01[
            0
        ]

        c2 = maf_01[
            1
        ]

        maf_preview = maf_to_preview(
            real_x
        )

        fake_rgb = fake_y_to_rgb(
            fake_y
        )

        (
            hematoxylin,
            eosin,
        ) = deconvolve_he(
            fake_rgb
        )

        hematoxylin_display = (
            normalize_for_display(
                hematoxylin
            )
        )

        eosin_display = (
            normalize_for_display(
                eosin
            )
        )

        # -------------------------------------------------
        # Diagnostic correlations
        #
        # IMPORTANT:
        # These use the RAW stain concentration maps,
        # not display-normalized maps.
        # -------------------------------------------------

        corr_c0_h = pearson_corr(
            c0,
            hematoxylin,
        )

        corr_c0_e = pearson_corr(
            c0,
            eosin,
        )

        corr_c2_h = pearson_corr(
            c2,
            hematoxylin,
        )

        corr_c2_e = pearson_corr(
            c2,
            eosin,
        )

        print()
        print(
            f"Sample {sample_number:02d}"
        )

        print(
            f"  C0 ↔ H: {corr_c0_h:+.4f}"
        )

        print(
            f"  C0 ↔ E: {corr_c0_e:+.4f}"
        )

        print(
            f"  C2 ↔ H: {corr_c2_h:+.4f}"
        )

        print(
            f"  C2 ↔ E: {corr_c2_e:+.4f}"
        )

        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------

        correlations = {
            "c0_h":
                corr_c0_h,

            "c0_e":
                corr_c0_e,

            "c2_h":
                corr_c2_h,

            "c2_e":
                corr_c2_e,
        }


        montage = make_diagnostic_montage(
            maf_preview=
                maf_preview,

            c0=
                c0,

            c2=
                c2,

            fake_rgb=
                fake_rgb,

            hematoxylin_display=
                hematoxylin_display,

            eosin_display=
                eosin_display,

            correlations=
                correlations,
        )


        output_path = (
            OUTPUT_DIR
            / (
                f"sample_"
                f"{sample_number:02d}"
                f"_he_deconvolution.png"
            )
        )


        montage.save(
            output_path
        )


        print(
            "  Saved:",
            output_path,
        )

        print(
            "  Saved:",
            output_path,
        )

    print()
    print(
        "=" * 72
    )

    print(
        "H&E DECONVOLUTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
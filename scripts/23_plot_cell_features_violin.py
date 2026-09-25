from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

RESULT_DIR = (
    ROOT
    / "outputs"
    / "morphology_crosswave_vs_gt_fast"
    / "cell_features_fast"
)

CW_CSV = (
    RESULT_DIR
    / "crosswave_cell_features.csv"
)

GT_CSV = (
    RESULT_DIR
    / "ground_truth_cell_features.csv"
)

METRICS_CSV = (
    RESULT_DIR
    / "cell_features_metrics_overall.csv"
)

OUTPUT_DIR = (
    RESULT_DIR
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "cell_area",
    "cell_eccentricity",
    "cell_circularity",
    "nucleus_cell_area_ratio",
]


DISPLAY_NAMES = {
    "cell_area":
        "Cell: Area",

    "cell_eccentricity":
        "Cell: Eccentricity",

    "cell_circularity":
        "Cell: Circularity",

    "nucleus_cell_area_ratio":
        "Nucleus/Cell area ratio",
}


# ============================================================
# COLORS
# ============================================================

CW_COLOR = "#A9D3DF"
GT_COLOR = "#E8E19C"
EDGE_COLOR = "#777777"


# ============================================================
# SETTINGS
# ============================================================

LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5

MAX_KDE_SAMPLES = 50000
KDE_POINTS = 500
HALF_WIDTH = 0.43
RANDOM_SEED = 42


# ============================================================
# LOAD
# ============================================================

cw_df = pd.read_csv(
    CW_CSV
)

gt_df = pd.read_csv(
    GT_CSV
)

metrics_df = pd.read_csv(
    METRICS_CSV
)


# ============================================================
# HELPERS
# ============================================================

def values(
    df,
    feature,
):

    x = pd.to_numeric(
        df[
            feature
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return x[
        np.isfinite(
            x
        )
    ]


def overlap_for(
    feature,
):

    row = metrics_df[
        metrics_df[
            "feature"
        ] == feature
    ]

    if row.empty:
        return np.nan

    return float(
        row.iloc[0][
            "overlap_coefficient"
        ]
    )


def subsample(
    x,
    seed,
):

    if len(
        x
    ) <= MAX_KDE_SAMPLES:

        return x

    rng = np.random.default_rng(
        seed
    )

    idx = rng.choice(
        len(
            x
        ),
        size=
            MAX_KDE_SAMPLES,
        replace=
            False,
    )

    return x[
        idx
    ]


def draw_split_violin(
    ax,
    cw_values,
    gt_values,
    title,
    overlap,
    index,
):

    pooled = np.concatenate(
        [
            cw_values,
            gt_values,
        ]
    )

    y_min = float(
        np.percentile(
            pooled,
            LOW_PERCENTILE,
        )
    )

    y_max = float(
        np.percentile(
            pooled,
            HIGH_PERCENTILE,
        )
    )

    if y_max <= y_min:
        y_max = y_min + 1e-6


    cw_display = cw_values[
        (
            cw_values >= y_min
        )
        & (
            cw_values <= y_max
        )
    ]

    gt_display = gt_values[
        (
            gt_values >= y_min
        )
        & (
            gt_values <= y_max
        )
    ]


    cw_input = subsample(
        cw_display,
        RANDOM_SEED
        + index,
    )

    gt_input = subsample(
        gt_display,
        RANDOM_SEED
        + 100
        + index,
    )


    y_grid = np.linspace(
        y_min,
        y_max,
        KDE_POINTS,
    )


    cw_density = gaussian_kde(
        cw_input
    )(
        y_grid
    )

    gt_density = gaussian_kde(
        gt_input
    )(
        y_grid
    )


    max_density = max(
        float(
            np.max(
                cw_density
            )
        ),
        float(
            np.max(
                gt_density
            )
        ),
    )


    cw_width = (
        cw_density
        / max_density
        * HALF_WIDTH
    )

    gt_width = (
        gt_density
        / max_density
        * HALF_WIDTH
    )


    ax.fill_betweenx(
        y_grid,
        -cw_width,
        0,
        facecolor=
            CW_COLOR,
        edgecolor=
            EDGE_COLOR,
        linewidth=
            0.8,
    )

    ax.fill_betweenx(
        y_grid,
        0,
        gt_width,
        facecolor=
            GT_COLOR,
        edgecolor=
            EDGE_COLOR,
        linewidth=
            0.8,
    )


    # Quartiles
    cw_q = np.percentile(
        cw_values,
        [
            25,
            50,
            75,
        ],
    )

    gt_q = np.percentile(
        gt_values,
        [
            25,
            50,
            75,
        ],
    )


    for q in cw_q:

        if y_min <= q <= y_max:

            ax.hlines(
                q,
                -0.31,
                0,
                linestyle=
                    "--",
                linewidth=
                    0.7,
                color=
                    EDGE_COLOR,
            )


    for q in gt_q:

        if y_min <= q <= y_max:

            ax.hlines(
                q,
                0,
                0.31,
                linestyle=
                    "--",
                linewidth=
                    0.7,
                color=
                    EDGE_COLOR,
            )


    ax.axvline(
        0,
        color=
            EDGE_COLOR,
        linewidth=
            0.8,
    )


    ax.set_title(
        title,
        fontsize=
            11,
    )


    ax.text(
        0.96,
        0.95,
        f"Overlap = {overlap:.3f}",
        transform=
            ax.transAxes,
        ha=
            "right",
        va=
            "top",
        fontsize=
            9,
    )


    ax.set_xlim(
        -0.5,
        0.5,
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    ax.set_xticks(
        []
    )

    ax.tick_params(
        axis=
            "y",
        labelsize=
            8,
    )


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(
        12.5,
        3.8,
    ),
)


for index, feature in enumerate(
    FEATURES
):

    cw_values = values(
        cw_df,
        feature,
    )

    gt_values = values(
        gt_df,
        feature,
    )

    overlap = overlap_for(
        feature
    )

    draw_split_violin(
        axes[
            index
        ],
        cw_values,
        gt_values,
        DISPLAY_NAMES[
            feature
        ],
        overlap,
        index,
    )


# ============================================================
# LEGEND
# ============================================================

fig.legend(
    handles=[
        Patch(
            facecolor=
                CW_COLOR,
            edgecolor=
                EDGE_COLOR,
            label=
                "CrossWave virtual H&E",
        ),
        Patch(
            facecolor=
                GT_COLOR,
            edgecolor=
                EDGE_COLOR,
            label=
                "Ground-truth H&E",
        ),
    ],
    loc=
        "upper center",
    bbox_to_anchor=
        (
            0.5,
            0.98,
        ),
    ncol=
        2,
    frameon=
        False,
)


fig.suptitle(
    "Approximate cell morphology fidelity of CrossWave virtual H&E",
    fontsize=
        14,
    fontweight=
        "bold",
    y=
        1.04,
)


plt.subplots_adjust(
    left=
        0.06,
    right=
        0.99,
    bottom=
        0.12,
    top=
        0.78,
    wspace=
        0.28,
)


# ============================================================
# SAVE
# ============================================================

png_path = (
    OUTPUT_DIR
    / "cell_features_split_violin.png"
)

svg_path = (
    OUTPUT_DIR
    / "cell_features_split_violin.svg"
)


fig.savefig(
    png_path,
    dpi=
        600,
    bbox_inches=
        "tight",
)

fig.savefig(
    svg_path,
    bbox_inches=
        "tight",
)


print(
    png_path
)

print(
    svg_path
)


plt.show()
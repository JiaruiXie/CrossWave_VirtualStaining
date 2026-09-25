import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
    ),
)


from src.data.subeval_dataset import (
    SubEvalTileDataset,
)


def main():

    root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "Sub_eval"
    )

    af_dataset = (
        SubEvalTileDataset(
            root=root,
            domain="af",
            tile_size=256,
            stride=256,
        )
    )

    he_dataset = (
        SubEvalTileDataset(
            root=root,
            domain="he",
            tile_size=256,
            stride=256,
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "AF"
    )

    print(
        "=" * 70
    )

    item = af_dataset[
        0
    ]

    print(
        "shape:",
        tuple(
            item[
                "image"
            ].shape
        ),
    )

    print(
        "min:",
        float(
            item[
                "image"
            ].min()
        ),
    )

    print(
        "max:",
        float(
            item[
                "image"
            ].max()
        ),
    )

    print(
        "path:",
        item[
            "path"
        ],
    )

    print(
        "tile x/y:",
        item[
            "x"
        ],
        item[
            "y"
        ],
    )

    print()
    print(
        "=" * 70
    )

    print(
        "H&E"
    )

    print(
        "=" * 70
    )

    item = he_dataset[
        0
    ]

    print(
        "shape:",
        tuple(
            item[
                "image"
            ].shape
        ),
    )

    print(
        "min:",
        float(
            item[
                "image"
            ].min()
        ),
    )

    print(
        "max:",
        float(
            item[
                "image"
            ].max()
        ),
    )

    print(
        "path:",
        item[
            "path"
        ],
    )

    print()
    print(
        "AF tiles:",
        len(
            af_dataset
        ),
    )

    print(
        "H&E tiles:",
        len(
            he_dataset
        ),
    )

    assert (
        af_dataset[
            0
        ][
            "image"
        ].shape[0]
        == 2
    )

    assert (
        he_dataset[
            0
        ][
            "image"
        ].shape[0]
        == 3
    )

    print()
    print(
        "SUB_EVAL DATASET TEST PASSED"
    )


if __name__ == "__main__":
    main()
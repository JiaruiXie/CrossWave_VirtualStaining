from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.data.datasets import (
    ECycleGANDataset,
    CrossWaveDataset,
)


SPLIT_CSV = (
    ROOT
    / "data"
    / "metadata"
    / "data_split.csv"
)


def test_baseline(
    mode,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"Testing baseline: {mode}"
    )

    print(
        "=" * 72
    )

    dataset = (
        ECycleGANDataset(
            split_csv=SPLIT_CSV,
            split="train",
            mode=mode,
            augment=True,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    print(
        "X:",
        tuple(
            batch["x"].shape
        )
    )

    print(
        "Y:",
        tuple(
            batch["y"].shape
        )
    )

    print(
        "X range:",
        float(
            batch["x"].min()
        ),
        float(
            batch["x"].max()
        ),
    )

    print(
        "Y range:",
        float(
            batch["y"].min()
        ),
        float(
            batch["y"].max()
        ),
    )

    if mode == "maf":

        assert (
            batch["x"].shape
            == (4, 2, 256, 256)
        )

    else:

        assert (
            batch["x"].shape
            == (4, 1, 256, 256)
        )

    assert (
        batch["y"].shape
        == (4, 3, 256, 256)
    )


def test_crosswave():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "Testing CrossWave"
    )

    print(
        "=" * 72
    )

    dataset = (
        CrossWaveDataset(
            split_csv=SPLIT_CSV,
            split="train",
            augment=True,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    print(
        "MAF:",
        tuple(
            batch["maf"].shape
        )
    )

    print(
        "DAPI:",
        tuple(
            batch["dapi"].shape
        )
    )

    print(
        "H&E:",
        tuple(
            batch["he"].shape
        )
    )

    assert (
        batch["maf"].shape
        == (4, 2, 256, 256)
    )

    assert (
        batch["dapi"].shape
        == (4, 1, 256, 256)
    )

    assert (
        batch["he"].shape
        == (4, 3, 256, 256)
    )

    # -----------------------------------------------------
    # Because DAPI corresponds approximately to the
    # blue component of MAF, their alignment should still
    # be visible after paired augmentation.
    #
    # This is NOT intended as a strict equality test.
    # -----------------------------------------------------

    maf_dapi_channel = (
        batch["maf"][:, 1:2]
    )

    dapi = (
        batch["dapi"]
    )

    a = maf_dapi_channel.flatten(
        start_dim=1
    )

    b = dapi.flatten(
        start_dim=1
    )

    a = (
        a
        - a.mean(
            dim=1,
            keepdim=True,
        )
    )

    b = (
        b
        - b.mean(
            dim=1,
            keepdim=True,
        )
    )

    numerator = (
        a * b
    ).mean(
        dim=1
    )

    denominator = (
        a.std(
            dim=1
        )
        * b.std(
            dim=1
        )
        + 1e-8
    )

    ncc = (
        numerator
        / denominator
    )

    print(
        "MAF-blue ↔ DAPI NCC:",
        ncc.tolist(),
    )

    assert (
        torch.all(
            ncc > 0.90
        )
    )

    print(
        "\nCrossWave paired "
        "augmentation check PASSED."
    )


def main():

    test_baseline(
        "maf"
    )

    test_baseline(
        "dapi"
    )

    test_crosswave()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALL DATASET TESTS PASSED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
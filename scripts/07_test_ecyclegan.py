from pathlib import Path
import sys

import torch


ROOT = Path(
    r"C:\Users\jiaru\Documents\code_temp_files\CrossWave_VirtualStaining"
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.models.ecyclegan import (
    ECycleGAN
)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


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

    # =====================================================
    # MAF teacher configuration
    #
    # X = 3-channel released MAF
    # Y = 3-channel H&E
    # =====================================================

    model = ECycleGAN(
        x_channels=3,
        y_channels=3,
        num_res_blocks=9,
    ).to(device)

    x = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    y = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    # =====================================================
    # Generator forward
    # =====================================================

    with torch.no_grad():

        outputs = model(
            x,
            y,
            return_features=True,
        )

        pred_fake_x = model.D_X(
            outputs["fake_x"]
        )

        pred_fake_y = model.D_Y(
            outputs["fake_y"]
        )

    print("\nInput shapes")

    print(
        "X:",
        tuple(x.shape),
    )

    print(
        "Y:",
        tuple(y.shape),
    )

    print("\nGenerated images")

    print(
        "fake Y:",
        tuple(
            outputs["fake_y"].shape
        ),
    )

    print(
        "fake X:",
        tuple(
            outputs["fake_x"].shape
        ),
    )

    print("\nCycle reconstructions")

    print(
        "reconstructed X:",
        tuple(
            outputs[
                "reconstructed_x"
            ].shape
        ),
    )

    print(
        "reconstructed Y:",
        tuple(
            outputs[
                "reconstructed_y"
            ].shape
        ),
    )

    print("\nIntermediate features")

    print(
        "X -> Y features:",
        tuple(
            outputs[
                "features_xy"
            ].shape
        ),
    )

    print(
        "Y -> X features:",
        tuple(
            outputs[
                "features_yx"
            ].shape
        ),
    )

    print("\nDiscriminator outputs")

    print(
        "D_X(fake X):",
        tuple(
            pred_fake_x.shape
        ),
    )

    print(
        "D_Y(fake Y):",
        tuple(
            pred_fake_y.shape
        ),
    )

    print(
        "\nTotal parameters:",
        f"{count_parameters(model):,}",
    )

    # =====================================================
    # Assertions
    # =====================================================

    assert (
        outputs["fake_y"].shape
        == (1, 3, 256, 256)
    )

    assert (
        outputs["fake_x"].shape
        == (1, 3, 256, 256)
    )

    assert (
        outputs["reconstructed_x"].shape
        == (1, 3, 256, 256)
    )

    assert (
        outputs["reconstructed_y"].shape
        == (1, 3, 256, 256)
    )

    assert (
        outputs["features_xy"].shape
        == (1, 256, 64, 64)
    )

    assert (
        outputs["features_yx"].shape
        == (1, 256, 64, 64)
    )

    assert (
        pred_fake_x.shape
        == (1, 1, 30, 30)
    )

    assert (
        pred_fake_y.shape
        == (1, 1, 30, 30)
    )

    print(
        "\nE-CycleGAN structure test PASSED."
    )


if __name__ == "__main__":
    main()
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

from src.losses.ecyclegan_loss import (
    ECycleGANLoss
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
    # Model
    # =====================================================

    model = ECycleGAN(
        x_channels=3,
        y_channels=3,
        num_res_blocks=9,
    ).to(device)

    criterion = ECycleGANLoss(
        lambda_cyc=10.0,
        lambda_feat=1.0,
        lambda_sal=1.0,
    ).to(device)

    # =====================================================
    # Dummy input
    # =====================================================

    real_x = (
        torch.rand(
            1,
            3,
            256,
            256,
            device=device,
        )
        * 2
        - 1
    )

    real_y = (
        torch.rand(
            1,
            3,
            256,
            256,
            device=device,
        )
        * 2
        - 1
    )

    # =====================================================
    # Generator forward
    # =====================================================

    outputs = model(
        real_x,
        real_y,
        return_features=False,
    )

    fake_y = (
        outputs["fake_y"]
    )

    fake_x = (
        outputs["fake_x"]
    )

    reconstructed_x = (
        outputs[
            "reconstructed_x"
        ]
    )

    reconstructed_y = (
        outputs[
            "reconstructed_y"
        ]
    )

    # =====================================================
    # Predictions for generator adversarial loss
    # =====================================================

    pred_fake_y = (
        model.D_Y(
            fake_y
        )
    )

    pred_fake_x = (
        model.D_X(
            fake_x
        )
    )

    # =====================================================
    # Combined generator loss
    # =====================================================

    losses = (
        criterion.generator_loss(
            real_x=real_x,
            real_y=real_y,

            fake_x=fake_x,
            fake_y=fake_y,

            reconstructed_x=
                reconstructed_x,

            reconstructed_y=
                reconstructed_y,

            pred_fake_x=
                pred_fake_x,

            pred_fake_y=
                pred_fake_y,
        )
    )

    print(
        "\nGenerator losses"
    )

    for key, value in (
        losses.items()
    ):

        print(
            f"{key:>15}: "
            f"{float(value.detach()):.6f}"
        )

    # =====================================================
    # Test backward pass
    # =====================================================

    model.zero_grad(
        set_to_none=True
    )

    losses[
        "total"
    ].backward()

    # Check gradient in first forward-generator layer
    first_parameter = next(
        model.G_XY.parameters()
    )

    gradient_mean = (
        first_parameter.grad
        .abs()
        .mean()
        .item()
    )

    gradient_max = (
        first_parameter.grad
        .abs()
        .max()
        .item()
    )

    print(
        "\nG_XY gradient mean:",
        gradient_mean,
    )

    print(
        "G_XY gradient max:",
        gradient_max,
    )

    assert (
        gradient_max > 0
    )

    # =====================================================
    # Discriminator loss test
    # =====================================================

    pred_real_y = (
        model.D_Y(
            real_y
        )
    )

    pred_fake_y_detached = (
        model.D_Y(
            fake_y.detach()
        )
    )

    d_y = (
        criterion.discriminator_loss(
            pred_real=pred_real_y,
            pred_fake=
                pred_fake_y_detached,
        )
    )

    print(
        "\nD_Y loss"
    )

    print(
        "Total:",
        float(
            d_y["total"].detach()
        )
    )

    print(
        "Real:",
        float(
            d_y["real"].detach()
        )
    )

    print(
        "Fake:",
        float(
            d_y["fake"].detach()
        )
    )

    print(
        "\nCombined E-CycleGAN "
        "loss test PASSED."
    )


if __name__ == "__main__":
    main()
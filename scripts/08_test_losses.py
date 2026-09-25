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


from src.losses.gan_loss import GANLoss
from src.losses.cycle_loss import (
    CycleConsistencyLoss
)
from src.losses.perceptual_loss import (
    GlobalFeatureLoss
)
from src.losses.saliency_loss import (
    SaliencyLoss
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

    # -----------------------------------------------------
    # GAN loss
    # -----------------------------------------------------

    gan_loss = GANLoss().to(
        device
    )

    prediction = torch.randn(
        2,
        1,
        30,
        30,
        device=device,
    )

    loss_real = gan_loss(
        prediction,
        True,
    )

    loss_fake = gan_loss(
        prediction,
        False,
    )

    print(
        "\nGAN loss"
    )

    print(
        "Real:",
        float(loss_real)
    )

    print(
        "Fake:",
        float(loss_fake)
    )

    # -----------------------------------------------------
    # Cycle loss
    # -----------------------------------------------------

    cycle_loss = (
        CycleConsistencyLoss()
        .to(device)
    )

    real_x = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    reconstructed_x = (
        real_x
        + 0.1
        * torch.randn_like(
            real_x
        )
    )

    real_y = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    reconstructed_y = (
        real_y
        + 0.1
        * torch.randn_like(
            real_y
        )
    )

    cycle = cycle_loss(
        real_x,
        reconstructed_x,
        real_y,
        reconstructed_y,
    )

    print(
        "\nCycle loss"
    )

    print(
        "Total:",
        float(cycle["total"])
    )

    print(
        "X:",
        float(cycle["x"])
    )

    print(
        "Y:",
        float(cycle["y"])
    )

    # -----------------------------------------------------
    # VGG feature loss
    # -----------------------------------------------------

    feature_loss = (
        GlobalFeatureLoss()
        .to(device)
    )

    generated = (
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

    target = (
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

    feature = feature_loss(
        generated,
        target,
    )

    print(
        "\nFeature loss"
    )

    print(
        "Total:",
        float(feature["total"])
    )

    print(
        "Layer losses:"
    )

    for i, value in enumerate(
        feature["layers"],
        start=1,
    ):

        print(
            f"  Layer {i}: "
            f"{float(value):.6f}"
        )

    # -----------------------------------------------------
    # Saliency loss
    # -----------------------------------------------------

    saliency_loss = (
        SaliencyLoss(
            threshold=170,
            kernel_size=3,
            sharpness=20.0,
        )
        .to(device)
    )

    generated_sal = (
        torch.rand(
            1,
            3,
            256,
            256,
            device=device,
            requires_grad=True,
        )
        * 2
        - 1
    )

    target_sal = (
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

    # generated_sal is not a leaf after *2-1,
    # so explicitly retain its gradient.
    generated_sal.retain_grad()

    saliency = saliency_loss(
        generated_sal,
        target_sal,
    )

    saliency.backward()

    gradient_mean = (
        generated_sal.grad
        .abs()
        .mean()
        .item()
    )

    gradient_max = (
        generated_sal.grad
        .abs()
        .max()
        .item()
    )

    print(
        "\nSaliency loss:",
        float(
            saliency.detach()
        )
    )

    print(
        "Saliency gradient mean:",
        gradient_mean,
    )

    print(
        "Saliency gradient max:",
        gradient_max,
    )

    assert (
        gradient_max > 0
    ), "Saliency loss has no gradient."

    print(
        "\nAll loss tests PASSED."
    )


if __name__ == "__main__":
    main()
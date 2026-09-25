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


from src.models.discriminator import (
    PatchGANDiscriminator
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

    discriminator = (
        PatchGANDiscriminator(
            input_channels=3,
        ).to(device)
    )

    x = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():

        output = (
            discriminator(x)
        )

    print(
        "Input:",
        tuple(x.shape),
    )

    print(
        "Output:",
        tuple(output.shape),
    )

    print(
        "Parameters:",
        f"{count_parameters(discriminator):,}",
    )

    assert (
        output.shape
        == (2, 1, 30, 30)
    )

    print(
        "\nPatchGAN test PASSED."
    )


if __name__ == "__main__":
    main()
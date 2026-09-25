import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN-style discriminator.

    For a 256x256 input, the output map should be:
        (B, 1, 30, 30)
    """

    def __init__(
        self,
        input_channels=3,
        base_channels=64,
    ):
        super().__init__()

        # -------------------------------------------------
        # Layer 1
        #
        # 256 -> 128
        # No normalization in first discriminator layer
        # -------------------------------------------------

        self.layer1 = nn.Sequential(

            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        # -------------------------------------------------
        # Layer 2
        #
        # 128 -> 64
        # -------------------------------------------------

        self.layer2 = nn.Sequential(

            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm2d(
                base_channels * 2,
                affine=False,
                track_running_stats=False,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        # -------------------------------------------------
        # Layer 3
        #
        # 64 -> 32
        # -------------------------------------------------

        self.layer3 = nn.Sequential(

            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm2d(
                base_channels * 4,
                affine=False,
                track_running_stats=False,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        # -------------------------------------------------
        # Layer 4
        #
        # 32 -> 31
        #
        # stride=1
        # -------------------------------------------------

        self.layer4 = nn.Sequential(

            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm2d(
                base_channels * 8,
                affine=False,
                track_running_stats=False,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        # -------------------------------------------------
        # Output
        #
        # 31 -> 30
        # -------------------------------------------------

        self.output_layer = nn.Conv2d(
            base_channels * 8,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return self.output_layer(x)
import torch
import torch.nn as nn


# =========================================================
# Residual block
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        padding_type="reflect",
    ):
        super().__init__()

        layers = []

        # -------------------------------------------------
        # First convolution
        # -------------------------------------------------

        if padding_type == "reflect":
            layers.append(
                nn.ReflectionPad2d(1)
            )
            padding = 0

        elif padding_type == "zero":
            padding = 1

        else:
            raise ValueError(
                f"Unsupported padding type: "
                f"{padding_type}"
            )

        layers.extend([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                bias=False,
            ),

            nn.InstanceNorm2d(
                channels,
                affine=False,
                track_running_stats=False,
            ),

            nn.ReLU(
                inplace=True
            ),
        ])

        # -------------------------------------------------
        # Second convolution
        # -------------------------------------------------

        if padding_type == "reflect":
            layers.append(
                nn.ReflectionPad2d(1)
            )
            padding = 0

        else:
            padding = 1

        layers.extend([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                bias=False,
            ),

            nn.InstanceNorm2d(
                channels,
                affine=False,
                track_running_stats=False,
            ),
        ])

        self.block = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return (
            x + self.block(x)
        )


# =========================================================
# ResNet-9 Generator
# =========================================================

class ResNetGenerator(nn.Module):

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        base_channels=64,
        num_res_blocks=9,
        padding_type="reflect",
    ):
        super().__init__()

        self.input_channels = (
            input_channels
        )

        self.output_channels = (
            output_channels
        )

        self.base_channels = (
            base_channels
        )

        self.num_res_blocks = (
            num_res_blocks
        )

        # =================================================
        # Encoder
        #
        # 256 x 256
        #    ↓
        # 128 x 128
        #    ↓
        # 64 x 64
        # =================================================

        if padding_type == "reflect":

            self.initial = nn.Sequential(

                nn.ReflectionPad2d(3),

                nn.Conv2d(
                    input_channels,
                    base_channels,
                    kernel_size=7,
                    stride=1,
                    padding=0,
                    bias=False,
                ),

                nn.InstanceNorm2d(
                    base_channels,
                    affine=False,
                    track_running_stats=False,
                ),

                nn.ReLU(
                    inplace=True
                ),
            )

        else:

            self.initial = nn.Sequential(

                nn.Conv2d(
                    input_channels,
                    base_channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                ),

                nn.InstanceNorm2d(
                    base_channels,
                    affine=False,
                    track_running_stats=False,
                ),

                nn.ReLU(
                    inplace=True
                ),
            )

        self.down1 = nn.Sequential(

            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm2d(
                base_channels * 2,
                affine=False,
                track_running_stats=False,
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self.down2 = nn.Sequential(

            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm2d(
                base_channels * 4,
                affine=False,
                track_running_stats=False,
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        # =================================================
        # Residual trunk
        #
        # Feature shape:
        # 256 x 64 x 64
        # =================================================

        self.res_blocks = nn.Sequential(

            *[
                ResidualBlock(
                    base_channels * 4,
                    padding_type=padding_type,
                )
                for _ in range(
                    num_res_blocks
                )
            ]
        )

        # =================================================
        # Decoder
        # =================================================

        self.up1 = nn.Sequential(

            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),

            nn.InstanceNorm2d(
                base_channels * 2,
                affine=False,
                track_running_stats=False,
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self.up2 = nn.Sequential(

            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),

            nn.InstanceNorm2d(
                base_channels,
                affine=False,
                track_running_stats=False,
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        if padding_type == "reflect":

            self.output_layer = nn.Sequential(

                nn.ReflectionPad2d(3),

                nn.Conv2d(
                    base_channels,
                    output_channels,
                    kernel_size=7,
                    stride=1,
                    padding=0,
                ),

                nn.Tanh(),
            )

        else:

            self.output_layer = nn.Sequential(

                nn.Conv2d(
                    base_channels,
                    output_channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                ),

                nn.Tanh(),
            )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        x,
        return_features=False,
    ):

        x = self.initial(x)

        x = self.down1(x)

        x = self.down2(x)

        features = (
            self.res_blocks(x)
        )

        x = self.up1(
            features
        )

        x = self.up2(x)

        output = (
            self.output_layer(x)
        )

        if return_features:

            return (
                output,
                features,
            )

        return output
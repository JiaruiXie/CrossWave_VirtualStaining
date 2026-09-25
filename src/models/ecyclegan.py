import torch
import torch.nn as nn

from src.models.generator import ResNetGenerator
from src.models.discriminator import PatchGANDiscriminator


class ECycleGAN(nn.Module):
    """
    Bidirectional E-CycleGAN model.

    Domain X:
        MAF or DAPI-filter AF

    Domain Y:
        H&E

    Contains:
        G_XY : X -> Y
        G_YX : Y -> X
        D_X  : discriminate real/fake X
        D_Y  : discriminate real/fake Y
    """

    def __init__(
        self,
        x_channels=3,
        y_channels=3,
        base_channels=64,
        num_res_blocks=9,
        padding_type="reflect",
    ):
        super().__init__()

        # =================================================
        # Generators
        # =================================================

        self.G_XY = ResNetGenerator(
            input_channels=x_channels,
            output_channels=y_channels,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            padding_type=padding_type,
        )

        self.G_YX = ResNetGenerator(
            input_channels=y_channels,
            output_channels=x_channels,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            padding_type=padding_type,
        )

        # =================================================
        # Discriminators
        # =================================================

        self.D_X = PatchGANDiscriminator(
            input_channels=x_channels,
            base_channels=base_channels,
        )

        self.D_Y = PatchGANDiscriminator(
            input_channels=y_channels,
            base_channels=base_channels,
        )

    # =====================================================
    # Translation helpers
    # =====================================================

    def x_to_y(
        self,
        x,
        return_features=False,
    ):

        return self.G_XY(
            x,
            return_features=return_features,
        )

    def y_to_x(
        self,
        y,
        return_features=False,
    ):

        return self.G_YX(
            y,
            return_features=return_features,
        )

    # =====================================================
    # Cycle helpers
    # =====================================================

    def cycle_x(self, x):

        fake_y = self.G_XY(x)

        reconstructed_x = self.G_YX(
            fake_y
        )

        return (
            fake_y,
            reconstructed_x,
        )

    def cycle_y(self, y):

        fake_x = self.G_YX(y)

        reconstructed_y = self.G_XY(
            fake_x
        )

        return (
            fake_x,
            reconstructed_y,
        )

    # =====================================================
    # Full generator forward
    # =====================================================

    def forward(
        self,
        x,
        y,
        return_features=False,
    ):
        """
        Compute both translations and both cycles.

        This does NOT run discriminators.

        Returns a dictionary to keep training code explicit.
        """

        if return_features:

            fake_y, feat_xy = self.G_XY(
                x,
                return_features=True,
            )

            fake_x, feat_yx = self.G_YX(
                y,
                return_features=True,
            )

        else:

            fake_y = self.G_XY(x)
            fake_x = self.G_YX(y)

            feat_xy = None
            feat_yx = None

        reconstructed_x = self.G_YX(
            fake_y
        )

        reconstructed_y = self.G_XY(
            fake_x
        )

        return {
            "fake_y": fake_y,
            "fake_x": fake_x,

            "reconstructed_x": reconstructed_x,
            "reconstructed_y": reconstructed_y,

            "features_xy": feat_xy,
            "features_yx": feat_yx,
        }
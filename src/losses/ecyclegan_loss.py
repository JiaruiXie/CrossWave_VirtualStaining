import torch.nn as nn

from src.losses.gan_loss import (
    GANLoss
)

from src.losses.cycle_loss import (
    CycleConsistencyLoss
)

from src.losses.perceptual_loss import (
    GlobalFeatureLoss
)

from src.losses.saliency_loss import (
    SaliencyLoss
)


class ECycleGANLoss(nn.Module):

    def __init__(
        self,
        lambda_cyc=10.0,
        lambda_feat=1.0,
        lambda_sal=1.0,
        saliency_threshold=170,
        saliency_sharpness=20.0,
        auxiliary_invert_x=False,
    ):
        """
        Combined E-CycleGAN loss.

        Generator objective:

            L_G =
                L_GAN_XY
                + L_GAN_YX
                + lambda_cyc  * L_cycle
                + lambda_feat * L_feature
                + lambda_sal  * L_saliency

        auxiliary_invert_x:
            If True, ONLY the X-domain representation used
            inside the feature and saliency losses is
            contrast-inverted.

            X domain:
                1-channel DAPI-filter AF
                2-channel MAF [C0, C2]

            Y domain:
                3-channel H&E

            The transformation is:

                x_aux = -x

            for X-domain inputs normalized to [-1, 1].

            This does NOT modify:
                - generator input
                - generated images
                - adversarial loss
                - cycle loss
                - discriminator loss
                - FID/KID evaluation
        """

        super().__init__()

        # =================================================
        # Loss weights
        # =================================================

        self.lambda_cyc = float(
            lambda_cyc
        )

        self.lambda_feat = float(
            lambda_feat
        )

        self.lambda_sal = float(
            lambda_sal
        )

        self.auxiliary_invert_x = bool(
            auxiliary_invert_x
        )

        # =================================================
        # Adversarial loss
        # =================================================

        self.gan_loss = (
            GANLoss()
        )

        # =================================================
        # Cycle consistency
        # =================================================

        self.cycle_loss = (
            CycleConsistencyLoss()
        )

        # =================================================
        # Global feature preservation
        #
        # X-domain contrast inversion, when enabled,
        # happens internally inside GlobalFeatureLoss.
        # =================================================

        self.feature_loss = (
            GlobalFeatureLoss(
                auxiliary_invert_x=
                    self.auxiliary_invert_x
            )
        )

        # =================================================
        # Saliency preservation
        #
        # X-domain contrast inversion, when enabled,
        # happens internally inside SaliencyLoss.
        # =================================================

        self.saliency_loss = (
            SaliencyLoss(
                threshold=
                    saliency_threshold,

                kernel_size=3,

                sharpness=
                    saliency_sharpness,

                auxiliary_invert_x=
                    self.auxiliary_invert_x,
            )
        )

    # =====================================================
    # Generator objective
    # =====================================================

    def generator_loss(
        self,
        real_x,
        real_y,
        fake_x,
        fake_y,
        reconstructed_x,
        reconstructed_y,
        pred_fake_x,
        pred_fake_y,
    ):

        # -------------------------------------------------
        # Adversarial
        #
        # X -> Y:
        #     fake_y should look like real H&E
        #
        # Y -> X:
        #     fake_x should look like real AF
        #
        # IMPORTANT:
        # No contrast inversion is applied here.
        # -------------------------------------------------

        loss_gan_xy = (
            self.gan_loss(
                pred_fake_y,
                True,
            )
        )

        loss_gan_yx = (
            self.gan_loss(
                pred_fake_x,
                True,
            )
        )

        # -------------------------------------------------
        # Cycle consistency
        #
        # real_x
        #     -> fake_y
        #     -> reconstructed_x
        #
        # real_y
        #     -> fake_x
        #     -> reconstructed_y
        #
        # IMPORTANT:
        # Cycle consistency remains in the original
        # physical image representation.
        # -------------------------------------------------

        cycle = (
            self.cycle_loss(
                real_x,
                reconstructed_x,
                real_y,
                reconstructed_y,
            )
        )

        loss_cycle = (
            cycle[
                "total"
            ]
        )

        # -------------------------------------------------
        # Global feature preservation
        #
        # Compare each source image with its OWN
        # translated image.
        #
        # X -> Y:
        #
        #     real_x
        #         vs
        #     fake_y
        #
        # If auxiliary_invert_x=True:
        #
        #     VGG(inverted real_x)
        #         vs
        #     VGG(fake_y)
        #
        #
        # Y -> X:
        #
        #     real_y
        #         vs
        #     fake_x
        #
        # If auxiliary_invert_x=True:
        #
        #     VGG(real_y)
        #         vs
        #     VGG(inverted fake_x)
        #
        # The inversion is handled internally by
        # GlobalFeatureLoss based on channel count.
        # -------------------------------------------------

        feat_xy = (
            self.feature_loss(
                translated=
                    fake_y,

                source=
                    real_x,
            )
        )

        feat_yx = (
            self.feature_loss(
                translated=
                    fake_x,

                source=
                    real_y,
            )
        )

        loss_feature = (
            feat_xy[
                "total"
            ]
            +
            feat_yx[
                "total"
            ]
        )

        # -------------------------------------------------
        # Saliency preservation
        #
        # Same source -> translated relationship.
        #
        # X -> Y:
        #
        #     saliency(fake_y)
        #         vs
        #     saliency(inverted real_x)
        #
        # Y -> X:
        #
        #     saliency(inverted fake_x)
        #         vs
        #     saliency(real_y)
        #
        # when auxiliary_invert_x=True.
        #
        # Inversion is handled internally by SaliencyLoss.
        # -------------------------------------------------

        loss_sal_xy = (
            self.saliency_loss(
                generated=
                    fake_y,

                target=
                    real_x,
            )
        )

        loss_sal_yx = (
            self.saliency_loss(
                generated=
                    fake_x,

                target=
                    real_y,
            )
        )

        loss_saliency = (
            loss_sal_xy
            +
            loss_sal_yx
        )

        # -------------------------------------------------
        # Full generator loss
        # -------------------------------------------------

        total = (
            loss_gan_xy
            +
            loss_gan_yx

            +
            self.lambda_cyc
            *
            loss_cycle

            +
            self.lambda_feat
            *
            loss_feature

            +
            self.lambda_sal
            *
            loss_saliency
        )

        # -------------------------------------------------
        # Return detailed terms for logging
        # -------------------------------------------------

        return {
            "total":
                total,

            "gan_xy":
                loss_gan_xy,

            "gan_yx":
                loss_gan_yx,

            "cycle":
                loss_cycle,

            "cycle_x":
                cycle[
                    "x"
                ],

            "cycle_y":
                cycle[
                    "y"
                ],

            "feature":
                loss_feature,

            "feature_xy":
                feat_xy[
                    "total"
                ],

            "feature_yx":
                feat_yx[
                    "total"
                ],

            "saliency":
                loss_saliency,

            "saliency_xy":
                loss_sal_xy,

            "saliency_yx":
                loss_sal_yx,
        }

    # =====================================================
    # Discriminator objective
    # =====================================================

    def discriminator_loss(
        self,
        pred_real,
        pred_fake,
    ):
        """
        Standard discriminator BCE objective.

        No auxiliary contrast inversion is used here.

        The discriminator must always distinguish real and
        generated samples in their original domain.
        """

        loss_real = (
            self.gan_loss(
                pred_real,
                True,
            )
        )

        loss_fake = (
            self.gan_loss(
                pred_fake,
                False,
            )
        )

        total = (
            0.5
            *
            (
                loss_real
                +
                loss_fake
            )
        )

        return {
            "total":
                total,

            "real":
                loss_real,

            "fake":
                loss_fake,
        }
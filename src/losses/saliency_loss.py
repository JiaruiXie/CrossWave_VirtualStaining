import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyLoss(nn.Module):
    """
    Differentiable approximation of the saliency-map loss
    described in the paper:

        RGB
          -> grayscale
          -> threshold at 170 / 255
          -> 3x3 opening
          -> 3x3 closing
          -> L1 difference

    A sigmoid soft threshold is used during training so
    gradients can propagate to the generator.

    Optional auxiliary X-domain contrast inversion:

        MAF / DAPI:
            x -> -x

    before saliency-map construction.

    This changes ONLY the representation used by the
    saliency loss. It does NOT change:

        - generator input
        - GAN loss
        - cycle loss
        - generated images
        - FID/KID evaluation
    """

    def __init__(
        self,
        threshold=170,
        kernel_size=3,
        sharpness=20.0,
        auxiliary_invert_x=False,
    ):
        super().__init__()

        self.threshold = (
            threshold / 255.0
        )

        self.kernel_size = (
            kernel_size
        )

        self.sharpness = (
            sharpness
        )

        self.auxiliary_invert_x = bool(
            auxiliary_invert_x
        )

        self.loss = nn.L1Loss()

    # =====================================================
    # X-domain auxiliary contrast transformation
    # =====================================================

    def prepare_auxiliary_input(
        self,
        x,
    ):
        """
        Optionally invert only X-domain representations.

        Domain identification by channel count:

            1 channel:
                DAPI-filter AF
                -> X domain

            2 channels:
                MAF [C0, C2]
                -> X domain

            3 channels:
                H&E
                -> Y domain

        Inputs are normalized to [-1, 1].

        Therefore contrast inversion is simply:

            x_inverted = -x

        Examples:

            normalized bright AF:
                +1 -> -1

            normalized dark AF:
                -1 -> +1

        H&E is never inverted.

        Important:
            The inversion happens BEFORE reconstruction
            of the missing MAF C1 channel.

            Therefore the intentionally empty C1 is still
            inserted as -1, rather than being accidentally
            inverted to +1.
        """

        if not self.auxiliary_invert_x:

            return x

        num_channels = x.shape[1]

        # -------------------------------------------------
        # X domain: DAPI or MAF
        # -------------------------------------------------

        if num_channels in {
            1,
            2,
        }:

            return -x

        # -------------------------------------------------
        # Y domain: H&E
        # -------------------------------------------------

        elif num_channels == 3:

            return x

        else:

            raise ValueError(
                "SaliencyLoss expects "
                "1-, 2-, or 3-channel input, "
                f"but got {num_channels} channels."
            )

    # =====================================================
    # RGB -> grayscale
    # =====================================================

    def rgb_to_gray(
        self,
        x,
    ):
        """
        Input:
            x in [-1, 1]

        Supports:

            1 channel:
                DAPI-filter AF

            2 channels:
                MAF [C0, C2]

            3 channels:
                RGB / H&E

        Output:
            grayscale in [0, 1]

        For 2-channel MAF:

            [C0, C2]
                ->
            [C0, empty C1, C2]

        where empty C1 = -1 in normalized space.
        """

        # -------------------------------------------------
        # Reconstruct released 3-channel MAF layout
        #
        # [C0, C2]
        # ->
        # [C0, empty C1, C2]
        # -------------------------------------------------

        if x.shape[1] == 2:

            c0 = x[:, 0:1]
            c2 = x[:, 1:2]

            # Original empty uint8 channel = 0.
            #
            # Under [-1, 1] normalization:
            #
            # 0 -> -1
            empty_c1 = torch.full_like(
                c0,
                -1.0,
            )

            x = torch.cat(
                [
                    c0,
                    empty_c1,
                    c2,
                ],
                dim=1,
            )

        # -------------------------------------------------
        # [-1, 1] -> [0, 1]
        # -------------------------------------------------

        x = (
            x + 1.0
        ) / 2.0

        x = torch.clamp(
            x,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Single-channel image
        # -------------------------------------------------

        if x.shape[1] == 1:

            return x

        # -------------------------------------------------
        # RGB-style grayscale
        # -------------------------------------------------

        if x.shape[1] != 3:

            raise ValueError(
                "SaliencyLoss expects "
                "1-, 2-, or 3-channel input."
            )

        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]

        return (
            0.299 * r
            + 0.587 * g
            + 0.114 * b
        )

    # =====================================================
    # Differentiable threshold
    # =====================================================

    def soft_threshold(
        self,
        x,
    ):
        """
        Differentiable approximation of:

            x >= threshold

        using a sigmoid.

        Paper threshold:
            170 / 255
        """

        return torch.sigmoid(
            self.sharpness
            * (
                x
                - self.threshold
            )
        )

    # =====================================================
    # Morphological operations
    # =====================================================

    def dilation(
        self,
        x,
    ):

        return F.max_pool2d(
            x,
            kernel_size=
                self.kernel_size,
            stride=1,
            padding=
                self.kernel_size // 2,
        )

    def erosion(
        self,
        x,
    ):

        return -F.max_pool2d(
            -x,
            kernel_size=
                self.kernel_size,
            stride=1,
            padding=
                self.kernel_size // 2,
        )

    def opening(
        self,
        x,
    ):

        return self.dilation(
            self.erosion(
                x
            )
        )

    def closing(
        self,
        x,
    ):

        return self.erosion(
            self.dilation(
                x
            )
        )

    # =====================================================
    # Saliency map
    # =====================================================

    def saliency_map(
        self,
        x,
    ):
        """
        Construct differentiable saliency map.

        Input must already have any desired
        X-domain contrast transformation applied.
        """

        gray = self.rgb_to_gray(
            x
        )

        saliency = self.soft_threshold(
            gray
        )

        saliency = self.opening(
            saliency
        )

        saliency = self.closing(
            saliency
        )

        return saliency

    # =====================================================
    # Loss
    # =====================================================

    def forward(
        self,
        generated,
        target,
    ):
        """
        Compare saliency maps of a translated image and
        its corresponding source image.

        With auxiliary_invert_x=False:

            behavior is identical to the original loss.

        With auxiliary_invert_x=True:

        ---------------------------------------------------
        Forward direction: MAF -> H&E
        ---------------------------------------------------

            generated:
                fake H&E
                3 channels
                -> unchanged

            target/source:
                real MAF
                2 channels
                -> contrast inverted

        So:

            saliency(fake H&E)

            vs

            saliency(inverted real MAF)


        ---------------------------------------------------
        Reverse direction: H&E -> MAF
        ---------------------------------------------------

            generated:
                fake MAF
                2 channels
                -> contrast inverted

            target/source:
                real H&E
                3 channels
                -> unchanged

        So:

            saliency(inverted fake MAF)

            vs

            saliency(real H&E)


        The X-domain transformation is therefore applied
        consistently regardless of translation direction.
        """

        # =================================================
        # Contrast-align X-domain representations
        # =================================================

        generated_for_saliency = (
            self.prepare_auxiliary_input(
                generated
            )
        )

        target_for_saliency = (
            self.prepare_auxiliary_input(
                target
            )
        )

        # =================================================
        # Generated map
        #
        # Keep gradient path to generator.
        # =================================================

        generated_map = (
            self.saliency_map(
                generated_for_saliency
            )
        )

        # =================================================
        # Target map
        #
        # No generator gradient required through target.
        # =================================================

        with torch.no_grad():

            target_map = (
                self.saliency_map(
                    target_for_saliency
                )
            )

        # =================================================
        # L1 saliency difference
        # =================================================

        return self.loss(
            generated_map,
            target_map,
        )
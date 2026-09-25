import torch
import torch.nn as nn

from torchvision.models import (
    vgg16,
    VGG16_Weights,
)


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        weights = (
            VGG16_Weights.IMAGENET1K_V1
        )

        vgg = vgg16(
            weights=weights
        ).features

        # relu1_1
        # relu2_1
        # relu3_1
        # relu4_1
        # relu5_1
        self.target_indices = {
            1,
            6,
            11,
            18,
            25,
        }

        self.features = vgg.eval()

        for parameter in (
            self.features.parameters()
        ):
            parameter.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

    def prepare_channels(
        self,
        x,
    ):
        """
        Convert input to 3 channels for VGG16.

        Supported inputs:

        1 channel:
            DAPI -> repeat to pseudo-RGB

        2 channels:
            MAF [C0, C2]
            -> reconstruct original released layout
            [C0, empty C1, C2]

        3 channels:
            already RGB / H&E

        Important:
            Any X-domain contrast inversion must happen
            BEFORE this function is called.

            This ensures the intentionally empty MAF C1
            channel remains empty (-1 in normalized space).
        """

        # -------------------------------------------------
        # Single-channel DAPI
        # -------------------------------------------------

        if x.shape[1] == 1:

            x = x.repeat(
                1,
                3,
                1,
                1,
            )

        # -------------------------------------------------
        # Two-channel MAF
        #
        # Input is normalized [-1, 1].
        #
        # Original empty uint8 channel = 0
        # therefore normalized value = -1.
        # -------------------------------------------------

        elif x.shape[1] == 2:

            c0 = x[:, 0:1]
            c2 = x[:, 1:2]

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
        # Normal 3-channel image
        # -------------------------------------------------

        elif x.shape[1] == 3:

            pass

        else:

            raise ValueError(
                "VGG16FeatureExtractor expects "
                "1-, 2-, or 3-channel input, "
                f"but got {x.shape[1]} channels."
            )

        return x

    def preprocess(
        self,
        x,
    ):
        """
        Expected generator/input range:
            [-1, 1]

        Convert to:
            [0, 1]

        Then apply ImageNet normalization.
        """

        x = self.prepare_channels(
            x
        )

        x = (
            x + 1.0
        ) / 2.0

        x = torch.clamp(
            x,
            0.0,
            1.0,
        )

        return (
            x - self.mean
        ) / self.std

    def forward(
        self,
        x,
    ):

        x = self.preprocess(
            x
        )

        outputs = []

        for index, layer in enumerate(
            self.features
        ):

            x = layer(x)

            if index in self.target_indices:

                outputs.append(
                    x
                )

            if len(outputs) == 5:
                break

        return outputs


class GlobalFeatureLoss(nn.Module):

    def __init__(
        self,
        auxiliary_invert_x=False,
    ):
        """
        Parameters
        ----------
        auxiliary_invert_x : bool
            If True, invert the contrast of X-domain
            representations before VGG feature extraction.

            X-domain is identified by channel count:

                1 channel -> DAPI-filter AF
                2 channels -> MAF [C0, C2]

            3-channel H&E images are never inverted.

            The inversion occurs in normalized [-1, 1]
            space:

                x_inverted = -x

            Therefore:

                bright AF -> dark auxiliary representation
                dark AF   -> bright auxiliary representation

            This affects ONLY the perceptual-loss
            representation. It does not modify generator
            input or generated images.
        """

        super().__init__()

        self.auxiliary_invert_x = bool(
            auxiliary_invert_x
        )

        self.extractor = (
            VGG16FeatureExtractor()
        )

        self.loss = nn.L1Loss()

    def prepare_auxiliary_input(
        self,
        x,
    ):
        """
        Apply contrast inversion only to X-domain images.

        X-domain:
            1 channel -> DAPI
            2 channels -> MAF

        Y-domain:
            3 channels -> H&E

        Inversion occurs BEFORE VGG channel reconstruction,
        which keeps the deliberately empty MAF green channel
        at its correct value.
        """

        if not self.auxiliary_invert_x:

            return x

        num_channels = x.shape[1]

        # -------------------------------------------------
        # X-domain representations
        # -------------------------------------------------

        if num_channels in {
            1,
            2,
        }:

            return -x

        # -------------------------------------------------
        # H&E remains unchanged
        # -------------------------------------------------

        elif num_channels == 3:

            return x

        else:

            raise ValueError(
                "GlobalFeatureLoss expects "
                "1-, 2-, or 3-channel input, "
                f"but got {num_channels} channels."
            )

    def forward(
        self,
        translated,
        source,
    ):
        """
        Compare VGG representations of:

            source image
                vs
            translated version of that same image

        Forward direction:

            source:
                MAF [C0, C2]

            translated:
                fake H&E

        With auxiliary_invert_x=True:

            VGG(inverted MAF)
                vs
            VGG(fake H&E)


        Reverse direction:

            source:
                real H&E

            translated:
                fake MAF

        With auxiliary_invert_x=True:

            VGG(real H&E)
                vs
            VGG(inverted fake MAF)


        Therefore the contrast transformation is applied
        consistently to the X domain regardless of whether
        X is the source or translated image.

        This is NOT comparison against an unrelated
        real H&E image.
        """

        # =================================================
        # Contrast-align X-domain representations
        # =================================================

        translated_for_feature = (
            self.prepare_auxiliary_input(
                translated
            )
        )

        source_for_feature = (
            self.prepare_auxiliary_input(
                source
            )
        )

        # =================================================
        # VGG feature extraction
        # =================================================

        translated_features = (
            self.extractor(
                translated_for_feature
            )
        )

        source_features = (
            self.extractor(
                source_for_feature
            )
        )

        # =================================================
        # Layer-wise L1 feature loss
        # =================================================

        layer_losses = []

        for (
            translated_feature,
            source_feature,
        ) in zip(
            translated_features,
            source_features,
        ):

            layer_loss = self.loss(
                translated_feature,
                source_feature,
            )

            layer_losses.append(
                layer_loss
            )

        total = (
            sum(
                layer_losses
            )
            / len(
                layer_losses
            )
        )

        return {
            "total":
                total,

            "layers":
                layer_losses,
        }
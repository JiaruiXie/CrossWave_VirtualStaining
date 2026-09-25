from pathlib import Path

import numpy as np
import torch
import tifffile


# =========================================================
# Tensor conversion
# =========================================================

def tensor_to_uint8(
    tensor,
):
    """
    Convert a normalized tensor from [-1, 1] to uint8.

    Input:
        C x H x W

    Output:
        H x W x C
        or H x W for single-channel images
    """

    if not torch.is_tensor(
        tensor
    ):
        raise TypeError(
            "Expected a torch.Tensor."
        )

    tensor = (
        tensor
        .detach()
        .cpu()
        .float()
    )

    if tensor.ndim != 3:
        raise ValueError(
            "Expected tensor shape C x H x W, "
            f"but got {tuple(tensor.shape)}"
        )

    # [-1, 1] -> [0, 1]
    tensor = (
        tensor + 1.0
    ) / 2.0

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0,
    )

    # [0, 1] -> [0, 255]
    array = (
        tensor.numpy()
        * 255.0
    )

    array = np.round(
        array
    ).astype(
        np.uint8
    )

    # C x H x W -> H x W x C
    array = np.transpose(
        array,
        (1, 2, 0),
    )

    # Single-channel:
    # H x W x 1 -> H x W
    if array.shape[2] == 1:
        array = array[
            ...,
            0
        ]

    return array


# =========================================================
# MAF visualization
# =========================================================

def maf_to_rgb(
    maf_tensor,
):
    """
    Convert 2-channel MAF tensor to RGB for visualization.

    Model representation:
        [C0, C2]

    Preview representation:
        R = C0
        G = 0
        B = C2

    Therefore:

        [C0, C2]
            ->
        [C0, 0, C2]

    Input:
        2 x H x W
        normalized to [-1, 1]

    Output:
        H x W x 3
        uint8
    """

    if not torch.is_tensor(
        maf_tensor
    ):
        raise TypeError(
            "Expected maf_tensor to be a torch.Tensor."
        )

    if maf_tensor.ndim != 3:
        raise ValueError(
            "Expected MAF tensor shape "
            "2 x H x W, "
            f"but got {tuple(maf_tensor.shape)}"
        )

    if maf_tensor.shape[0] != 2:
        raise ValueError(
            "MAF visualization expects exactly "
            "2 channels [C0, C2], "
            f"but got {maf_tensor.shape[0]}."
        )

    maf_tensor = (
        maf_tensor
        .detach()
        .cpu()
        .float()
    )

    # ---------------------------------------------
    # Convert the two real channels individually
    # from [-1, 1] to [0, 255].
    # ---------------------------------------------

    maf = (
        maf_tensor + 1.0
    ) / 2.0

    maf = torch.clamp(
        maf,
        0.0,
        1.0,
    )

    maf = (
        maf.numpy()
        * 255.0
    )

    maf = np.round(
        maf
    ).astype(
        np.uint8
    )

    c0 = maf[0]
    c2 = maf[1]

    # ---------------------------------------------
    # Empty green channel
    # ---------------------------------------------

    green = np.zeros_like(
        c0,
        dtype=np.uint8,
    )

    # H x W x 3
    rgb = np.stack(
        [
            c0,
            green,
            c2,
        ],
        axis=-1,
    )

    return rgb


# =========================================================
# Automatic preview conversion
# =========================================================

def tensor_to_preview(
    tensor,
    domain=None,
):
    """
    Convert tensor into a preview-friendly uint8 image.

    For MAF:
        2 channels -> RGB [C0, 0, C2]

    For H&E:
        3 channels -> normal RGB

    For DAPI:
        1 channel -> grayscale

    Parameters
    ----------
    tensor:
        C x H x W tensor in [-1, 1]

    domain:
        Optional string such as:
            "x"
            "maf"
            "y"
            "he"
            "dapi"

    Returns
    -------
    numpy.ndarray
    """

    if tensor.ndim != 3:
        raise ValueError(
            "Expected C x H x W tensor, "
            f"got {tuple(tensor.shape)}"
        )

    channels = (
        tensor.shape[0]
    )

    # ---------------------------------------------
    # Any 2-channel image in this project is MAF.
    # ---------------------------------------------

    if channels == 2:

        return maf_to_rgb(
            tensor
        )

    # ---------------------------------------------
    # Normal H&E / RGB
    # ---------------------------------------------

    if channels == 3:

        return tensor_to_uint8(
            tensor
        )

    # ---------------------------------------------
    # DAPI / grayscale
    # ---------------------------------------------

    if channels == 1:

        return tensor_to_uint8(
            tensor
        )

    raise ValueError(
        f"Unsupported preview channel count: {channels}"
    )


# =========================================================
# Save one image
# =========================================================

def save_tensor_image(
    tensor,
    output_path,
):
    """
    Save a tensor as a TIFF preview.

    Automatically converts:
        2-channel MAF -> RGB [C0, 0, C2]
        3-channel H&E -> RGB
        1-channel DAPI -> grayscale
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = tensor_to_preview(
        tensor
    )

    tifffile.imwrite(
        output_path,
        image,
    )


# =========================================================
# Save translation previews
# =========================================================

def save_translation_preview(
    model,
    x_batch,
    y_batch,
    output_dir,
    device,
    max_samples=4,
):
    """
    Save fixed validation examples for E-CycleGAN.

    For every sample:

        real_x
        fake_y
        reconstructed_x

        real_y
        fake_x
        reconstructed_y

    MAF-domain images are automatically rendered as:

        [C0, 0, C2]

    for convenient RGB visualization.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Support either raw tensors or dictionaries
    # returned by SingleDomainDataset loaders.
    # ---------------------------------------------

    if isinstance(
        x_batch,
        dict,
    ):
        real_x = x_batch[
            "image"
        ]
    else:
        real_x = x_batch

    if isinstance(
        y_batch,
        dict,
    ):
        real_y = y_batch[
            "image"
        ]
    else:
        real_y = y_batch

    real_x = real_x.to(
        device,
        non_blocking=True,
    )

    real_y = real_y.to(
        device,
        non_blocking=True,
    )

    # ---------------------------------------------
    # Preserve current train/eval state
    # ---------------------------------------------

    was_training = (
        model.training
    )

    model.eval()

    with torch.no_grad():

        # X -> Y
        fake_y = model.G_XY(
            real_x
        )

        # X -> Y -> X
        reconstructed_x = (
            model.G_YX(
                fake_y
            )
        )

        # Y -> X
        fake_x = model.G_YX(
            real_y
        )

        # Y -> X -> Y
        reconstructed_y = (
            model.G_XY(
                fake_x
            )
        )

    if was_training:
        model.train()

    # ---------------------------------------------
    # Number of examples
    # ---------------------------------------------

    n_x = min(
        max_samples,
        real_x.shape[0],
    )

    n_y = min(
        max_samples,
        real_y.shape[0],
    )

    n = min(
        n_x,
        n_y,
    )

    # ---------------------------------------------
    # Save individual TIFFs
    # ---------------------------------------------

    for i in range(
        n
    ):

        prefix = (
            f"sample_{i:02d}"
        )

        # =========================================
        # X-domain images
        # =========================================

        # Real MAF:
        # [C0, C2] -> [C0, 0, C2]
        save_tensor_image(
            real_x[i],
            output_dir
            / f"{prefix}_real_x.tif",
        )

        # Fake MAF generated from real H&E
        save_tensor_image(
            fake_x[i],
            output_dir
            / f"{prefix}_fake_x.tif",
        )

        # Reconstructed MAF
        save_tensor_image(
            reconstructed_x[i],
            output_dir
            / f"{prefix}_reconstructed_x.tif",
        )

        # =========================================
        # Y-domain images
        # =========================================

        # Generated H&E
        save_tensor_image(
            fake_y[i],
            output_dir
            / f"{prefix}_fake_y.tif",
        )

        # Real H&E
        save_tensor_image(
            real_y[i],
            output_dir
            / f"{prefix}_real_y.tif",
        )

        # Reconstructed H&E
        save_tensor_image(
            reconstructed_y[i],
            output_dir
            / f"{prefix}_reconstructed_y.tif",
        )

    print(
        f"Preview saved: "
        f"{output_dir}"
    )
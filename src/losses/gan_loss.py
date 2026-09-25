import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Original log-GAN objective.

    The paper writes the adversarial loss in
    binary cross-entropy form rather than LSGAN.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        prediction,
        target_is_real,
    ):

        target_value = (
            1.0 if target_is_real else 0.0
        )

        target = torch.full_like(
            prediction,
            fill_value=target_value,
        )

        return self.loss(
            prediction,
            target,
        )
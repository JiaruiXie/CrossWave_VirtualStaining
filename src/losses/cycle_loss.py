import torch.nn as nn


class CycleConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
        self,
        real_x,
        reconstructed_x,
        real_y,
        reconstructed_y,
    ):

        loss_x = self.loss(
            reconstructed_x,
            real_x,
        )

        loss_y = self.loss(
            reconstructed_y,
            real_y,
        )

        total = (
            loss_x
            + loss_y
        )

        return {
            "total": total,
            "x": loss_x,
            "y": loss_y,
        }
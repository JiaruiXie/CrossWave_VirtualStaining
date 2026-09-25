"""
DAPI-only baseline trainer for the current CrossWave virtual-staining study.

Purpose
-------
Train the same single-channel E-CycleGAN architecture used by the
CrossWave student, but WITHOUT:
    - teacher model
    - prediction distillation
    - feature distillation

For a fair comparison with the current CrossWave experiment, the baseline
uses the exact same single-channel tensor seen by the CrossWave student.

The existing MAF loader returns:
    X = [C0, C2]

This trainer slices:
    X_baseline = C2 only

The baseline therefore differs from CrossWave only in the absence of
teacher-derived losses.

It inherits the current adaptive E-CycleGAN trainer so it retains:
    - inverted-Y training
    - BCE adversarial loss
    - cycle / perceptual / saliency losses
    - replay buffers
    - cached real-H&E FID/KID features
    - FID-driven ReduceLROnPlateau
    - early stopping
    - best-FID checkpointing
    - checkpoint resume
"""

from __future__ import annotations

import copy
import time

import numpy as np

import torch

from src.training.trainer_ecyclegan import (
    ECycleGANTrainer,
)

from src.utils.visualization import (
    save_translation_preview,
)


class DAPIBaselineTrainer(
    ECycleGANTrainer
):
    """
    Single-band baseline using C2 only and no knowledge distillation.
    """

    def __init__(
        self,
        config,
        root,
    ):
        self.baseline_cfg = config.get(
            "baseline",
            {},
        )

        data_mode = str(
            config.get(
                "data",
                {},
            ).get(
                "mode",
                "maf",
            )
        ).lower()

        if data_mode != "maf":
            raise ValueError(
                "For direct comparison with the current CrossWave "
                "student, set data.mode='maf'. The loader returns "
                "[C0, C2], and this trainer slices C2 internally."
            )

        x_channels = int(
            config.get(
                "model",
                {},
            ).get(
                "x_channels",
                -1,
            )
        )

        if x_channels != 1:
            raise ValueError(
                "DAPI baseline requires model.x_channels=1."
            )

        super().__init__(
            config=config,
            root=root,
        )

        self.dapi_channel_index = int(
            self.baseline_cfg.get(
                "dapi_channel_index",
                1,
            )
        )

        if self.dapi_channel_index not in (
            0,
            1,
        ):
            raise ValueError(
                "The current MAF loader returns [C0, C2]. "
                "For the C2/DAPI input used by CrossWave, set "
                "baseline.dapi_channel_index=1."
            )

        print()
        print(
            "=" * 72
        )

        print(
            "DAPI-ONLY BASELINE"
        )

        print(
            "=" * 72
        )

        print(
            "Teacher:",
            "NONE",
        )

        print(
            "Knowledge distillation:",
            "NONE",
        )

        print(
            "Loader X:",
            "[C0, C2]",
        )

        print(
            "Baseline X:",
            (
                "C2 only"
                if self.dapi_channel_index == 1
                else "C0 only"
            ),
        )

        print(
            "Student/baseline input channels:",
            1,
        )

        print(
            "=" * 72
        )

        print()

    # =============================================================
    # Input conversion
    # =============================================================

    def _single_band_x(
        self,
        maf_x,
    ):
        """
        Slice one channel from the loader's [C0, C2] tensor.

        For the current CrossWave comparison:
            dapi_channel_index = 1
            -> C2 only
        """

        if maf_x.ndim != 4:
            raise ValueError(
                "Expected X tensor with shape [B,C,H,W], "
                f"got {tuple(maf_x.shape)}."
            )

        if maf_x.shape[1] != 2:
            raise ValueError(
                "Expected the MAF loader to return exactly "
                f"2 channels [C0,C2], got {maf_x.shape[1]}."
            )

        index = self.dapi_channel_index

        return maf_x[
            :,
            index:index + 1,
            :,
            :,
        ]

    # =============================================================
    # Train one epoch
    # =============================================================

    def train_epoch(
        self,
        epoch,
    ):
        self.model.train()

        start_time = time.time()

        running = {
            "G": 0.0,
            "D_X": 0.0,
            "D_Y": 0.0,
            "cycle": 0.0,
            "feature": 0.0,
            "saliency": 0.0,
        }

        print_every = int(
            self.logging_cfg.get(
                "print_every",
                100,
            )
        )

        total_batches = len(
            self.train_loader
        )

        for step, batch in enumerate(
            self.train_loader,
            start=1,
        ):
            maf_x = batch[
                "x"
            ].to(
                self.device,
                non_blocking=True,
            )

            real_x = self._single_band_x(
                maf_x
            )

            real_y = batch[
                "y"
            ].to(
                self.device,
                non_blocking=True,
            )

            real_y = (
                self._prepare_y_for_model(
                    real_y
                )
            )

            (
                loss_G,
                loss_cycle,
                loss_feature,
                loss_saliency,
                fake_x,
                fake_y,
            ) = self.generator_step(
                real_x,
                real_y,
            )

            pooled_fake_x = (
                self.fake_x_pool.query(
                    fake_x
                )
            )

            pooled_fake_y = (
                self.fake_y_pool.query(
                    fake_y
                )
            )

            loss_D_X = (
                self.discriminator_x_step(
                    real_x,
                    pooled_fake_x,
                )
            )

            loss_D_Y = (
                self.discriminator_y_step(
                    real_y,
                    pooled_fake_y,
                )
            )

            running["G"] += loss_G
            running["D_X"] += loss_D_X
            running["D_Y"] += loss_D_Y
            running["cycle"] += loss_cycle
            running["feature"] += loss_feature
            running["saliency"] += loss_saliency

            if (
                step == 1
                or step % print_every == 0
                or step == total_batches
            ):
                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"Epoch {epoch:03d} | "
                    f"Step {step:05d}/{total_batches:05d} | "
                    f"G {loss_G:.4f} | "
                    f"D_X {loss_D_X:.4f} | "
                    f"D_Y {loss_D_Y:.4f} | "
                    f"cyc {loss_cycle:.4f} | "
                    f"feat {loss_feature:.4f} | "
                    f"sal {loss_saliency:.4f} | "
                    f"{elapsed:.1f}s"
                )

        denominator = max(
            total_batches,
            1,
        )

        for key in running:
            running[key] /= denominator

        return running

    # =============================================================
    # Deterministic validation
    # =============================================================

    @torch.no_grad()
    def validate(
        self,
    ):
        self.model.eval()

        # ---------------------------------------------------------
        # X -> Y -> X
        # ---------------------------------------------------------
        x_total_loss = 0.0
        x_total_count = 0

        for batch in self.val_x_loader:
            maf_x = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
            )

            real_x = self._single_band_x(
                maf_x
            )

            fake_y = (
                self.model.G_XY(
                    real_x
                )
            )

            reconstructed_x = (
                self.model.G_YX(
                    fake_y
                )
            )

            loss = (
                torch.nn.functional.l1_loss(
                    reconstructed_x,
                    real_x,
                    reduction="mean",
                )
            )

            batch_size = int(
                real_x.shape[0]
            )

            x_total_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            x_total_count += batch_size

        val_cycle_x = (
            x_total_loss
            / max(
                x_total_count,
                1,
            )
        )

        # ---------------------------------------------------------
        # Y -> X -> Y
        # ---------------------------------------------------------
        y_total_loss = 0.0
        y_total_count = 0

        for batch in self.val_y_loader:
            real_y = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
            )

            real_y = (
                self._prepare_y_for_model(
                    real_y
                )
            )

            fake_x = (
                self.model.G_YX(
                    real_y
                )
            )

            reconstructed_y = (
                self.model.G_XY(
                    fake_x
                )
            )

            loss = (
                torch.nn.functional.l1_loss(
                    reconstructed_y,
                    real_y,
                    reduction="mean",
                )
            )

            batch_size = int(
                real_y.shape[0]
            )

            y_total_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            y_total_count += batch_size

        val_cycle_y = (
            y_total_loss
            / max(
                y_total_count,
                1,
            )
        )

        return {
            "val_cycle_x":
                val_cycle_x,
            "val_cycle_y":
                val_cycle_y,
            "val_cycle_mean":
                (
                    val_cycle_x
                    + val_cycle_y
                )
                / 2.0,
        }

    # =============================================================
    # FID / KID
    # =============================================================

    @torch.no_grad()
    def evaluate_fid_kid(
        self,
    ):
        """
        Evaluate the baseline using exactly one C2 input channel.

        Real H&E Inception features are reused from the existing cache.
        """

        if not self.fid_enabled:
            return {
                "fid": np.nan,
                "kid_mean": np.nan,
                "kid_std": np.nan,
            }

        self.model.eval()

        total_batches = len(
            self.metric_x_loader
        )

        print(
            "Evaluating DAPI-baseline FID/KID..."
        )

        for batch_idx, batch in enumerate(
            self.metric_x_loader
        ):
            maf_x = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
            )

            real_x = self._single_band_x(
                maf_x
            )

            fake_y = (
                self.model.G_XY(
                    real_x
                )
            )

            fake_y_physical = (
                self._restore_y_from_model(
                    fake_y
                )
            )

            fake_y_metric = (
                self._to_metric_image(
                    fake_y_physical
                )
            )

            self.fid_metric.update(
                fake_y_metric,
                real=False,
            )

            self.kid_metric.update(
                fake_y_metric,
                real=False,
            )

            if (
                batch_idx == 0
                or (
                    batch_idx + 1
                ) % 50 == 0
                or (
                    batch_idx + 1
                ) == total_batches
            ):
                print(
                    "  Fake metric batch "
                    f"{batch_idx + 1}/"
                    f"{total_batches}"
                )

        print(
            "  Computing FID..."
        )

        fid_score = (
            self.fid_metric
            .compute()
            .detach()
            .cpu()
            .item()
        )

        print(
            "  Computing KID..."
        )

        (
            kid_mean,
            kid_std,
        ) = self.kid_metric.compute()

        kid_mean = (
            kid_mean
            .detach()
            .cpu()
            .item()
        )

        kid_std = (
            kid_std
            .detach()
            .cpu()
            .item()
        )

        self.fid_metric.reset()
        self.kid_metric.reset()

        return {
            "fid":
                float(
                    fid_score
                ),
            "kid_mean":
                float(
                    kid_mean
                ),
            "kid_std":
                float(
                    kid_std
                ),
        }

    # =============================================================
    # Preview
    # =============================================================

    @torch.no_grad()
    def save_preview(
        self,
        epoch,
    ):
        """
        Save fixed validation previews using only the one-channel X input.
        Generated Y is restored to physical H&E before saving.
        """

        preview_dir = (
            self.preview_dir
            / f"epoch_{epoch:03d}"
        )

        x_batch = copy.copy(
            self.fixed_val_x
        )

        x_batch[
            "image"
        ] = self._single_band_x(
            self.fixed_val_x[
                "image"
            ]
        )

        trainer = self

        class PhysicalYGeneratorView(
            torch.nn.Module
        ):
            def __init__(
                self,
                model,
            ):
                super().__init__()

                self.model = model

            def G_XY(
                self,
                x,
            ):
                y_model = (
                    self.model.G_XY(
                        x
                    )
                )

                return (
                    trainer._restore_y_from_model(
                        y_model
                    )
                )

            def G_YX(
                self,
                y_physical,
            ):
                y_model = (
                    trainer._prepare_y_for_model(
                        y_physical
                    )
                )

                return (
                    self.model.G_YX(
                        y_model
                    )
                )

        preview_model = (
            PhysicalYGeneratorView(
                self.model
            )
        )

        save_translation_preview(
            model=
                preview_model,
            x_batch=
                x_batch,
            y_batch=
                self.fixed_val_y,
            output_dir=
                preview_dir,
            device=
                self.device,
            max_samples=4,
        )

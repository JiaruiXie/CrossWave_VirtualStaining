"""
CrossWave trainer for the breast virtual-staining case study.

Teacher
-------
Frozen 2-channel MAF E-CycleGAN teacher:
    X_teacher = [C0, C2]
    Y = inverted H&E model domain

Student
-------
Trainable single-band DAPI-filter E-CycleGAN student:
    X_student = C2 only
    Y = same inverted H&E model domain

The student keeps the same generator depth/capacity as the teacher except
for the first input convolution (1 channel instead of 2 channels).

CrossWave adds two teacher-student terms to the ordinary student
E-CycleGAN objective:

    L_total =
        L_ECycleGAN_student
        + lambda_kd_prediction * L1(Y_student, Y_teacher)
        + lambda_kd_feature    * L1(F_student, F_teacher)

where F is the representation after the residual stack in G_XY.

The MAF training loader is deliberately retained.  C2 is sliced from the
same augmented 2-channel MAF tensor used by the teacher, guaranteeing
pixel-perfect teacher/student input pairing without a second file lookup.

At inference only the single C2/DAPI-filter input is required.
"""

from __future__ import annotations

from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.ecyclegan import ECycleGAN
from src.training.trainer_ecyclegan import ECycleGANTrainer
from src.utils.visualization import save_translation_preview


class _FeatureCapture:
    """Small forward-hook helper used for teacher/student feature KD."""

    def __init__(self, module: torch.nn.Module, name: str):
        self.name = name
        self.output = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.output = output

    def clear(self):
        self.output = None

    def close(self):
        self.handle.remove()


def _find_distillation_module(generator: torch.nn.Module):
    """
    Find the representation immediately after the generator residual stack.

    The current project generator exposes ``residual_blocks``.  The fallbacks
    make this trainer tolerant to minor naming changes.
    """

    preferred_attributes = (
        "residual_blocks",
        "res_blocks",
        "resnet_blocks",
        "residuals",
    )

    for attribute in preferred_attributes:
        if hasattr(generator, attribute):
            module = getattr(generator, attribute)
            if isinstance(module, torch.nn.Module):
                return attribute, module

    candidates = []

    for name, module in generator.named_modules():
        class_name = module.__class__.__name__.lower()
        name_lower = name.lower()

        if (
            "residual" in class_name
            or "resnetblock" in class_name
            or "residual" in name_lower
            or "res_blocks" in name_lower
            or "resnet_blocks" in name_lower
        ):
            candidates.append((name, module))

    if candidates:
        return candidates[-1]

    raise RuntimeError(
        "Could not locate the generator residual representation for "
        "feature distillation. Expected an attribute such as "
        "'residual_blocks'."
    )


class CrossWaveTrainer(ECycleGANTrainer):
    """
    CrossWave = frozen multispectral teacher + single-band student.

    This class intentionally inherits the existing adaptive E-CycleGAN
    trainer so it reuses:

      - inverted-Y-domain handling
      - BCE adversarial / cycle / paper auxiliary losses
      - fake-image replay buffers
      - cached real-H&E FID/KID features
      - FID-driven ReduceLROnPlateau
      - early stopping
      - best-FID checkpointing
      - checkpoint resume support
    """

    def __init__(self, config, root):
        # -------------------------------------------------------------
        # Validate the CrossWave-specific configuration before the base
        # trainer constructs the student.
        # -------------------------------------------------------------
        self.crosswave_cfg = config.get("crosswave", {})

        if not self.crosswave_cfg:
            raise ValueError(
                "Missing top-level 'crosswave' configuration section."
            )

        data_mode = str(
            config.get("data", {}).get("mode", "maf")
        ).lower()

        if data_mode != "maf":
            raise ValueError(
                "CrossWave training must use data.mode='maf'. "
                "The loader must return the paired 2-channel [C0, C2] "
                "MAF tensor; the student C2 input is sliced from it."
            )

        student_x_channels = int(
            config.get("model", {}).get("x_channels", -1)
        )

        if student_x_channels != 1:
            raise ValueError(
                "CrossWave student requires model.x_channels=1."
            )

        super().__init__(
            config=config,
            root=root,
        )

        # -------------------------------------------------------------
        # CrossWave settings
        # -------------------------------------------------------------
        self.dapi_channel_index = int(
            self.crosswave_cfg.get(
                "dapi_channel_index",
                1,
            )
        )

        self.lambda_kd_prediction = float(
            self.crosswave_cfg.get(
                "lambda_kd_prediction",
                5.0,
            )
        )

        self.lambda_kd_feature = float(
            self.crosswave_cfg.get(
                "lambda_kd_feature",
                1.0,
            )
        )

        if self.dapi_channel_index not in (0, 1):
            raise ValueError(
                "The released MAF loader returns [C0, C2], so "
                "dapi_channel_index must be 0 or 1. For DAPI/C2 use 1."
            )

        # -------------------------------------------------------------
        # Load and freeze teacher
        # -------------------------------------------------------------
        teacher_checkpoint = Path(
            self.crosswave_cfg[
                "teacher_checkpoint"
            ]
        )

        if not teacher_checkpoint.is_absolute():
            teacher_checkpoint = (
                self.project_root
                / teacher_checkpoint
            )

        self.teacher_checkpoint = (
            teacher_checkpoint.resolve()
        )

        if not self.teacher_checkpoint.is_file():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: "
                f"{self.teacher_checkpoint}"
            )

        print()
        print("=" * 72)
        print("CROSSWAVE TEACHER")
        print("=" * 72)
        print(
            "Teacher checkpoint:",
            self.teacher_checkpoint,
        )

        teacher_checkpoint_payload = torch.load(
            self.teacher_checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        if (
            not isinstance(
                teacher_checkpoint_payload,
                dict,
            )
            or "model_state_dict"
            not in teacher_checkpoint_payload
        ):
            raise RuntimeError(
                "Teacher checkpoint must contain 'model_state_dict'."
            )

        teacher_config = teacher_checkpoint_payload.get(
            "config",
            {},
        )

        teacher_model_cfg = teacher_config.get(
            "model",
            {},
        )

        teacher_data_cfg = teacher_config.get(
            "data",
            {},
        )

        teacher_x_channels = int(
            teacher_model_cfg.get(
                "x_channels",
                2,
            )
        )

        teacher_y_channels = int(
            teacher_model_cfg.get(
                "y_channels",
                3,
            )
        )

        if teacher_x_channels != 2:
            raise RuntimeError(
                "CrossWave teacher must be the 2-channel MAF model. "
                f"Checkpoint reports x_channels={teacher_x_channels}."
            )

        teacher_invert_y = bool(
            teacher_data_cfg.get(
                "invert_y_domain",
                False,
            )
        )

        if (
            teacher_invert_y
            != self.invert_y_domain
        ):
            raise RuntimeError(
                "Teacher/student Y-domain mismatch. "
                f"teacher invert_y_domain={teacher_invert_y}, "
                f"student invert_y_domain={self.invert_y_domain}. "
                "Both must use the same model-domain H&E convention."
            )

        self.teacher = ECycleGAN(
            x_channels=
                teacher_x_channels,
            y_channels=
                teacher_y_channels,
            base_channels=
                int(
                    teacher_model_cfg.get(
                        "base_channels",
                        64,
                    )
                ),
            num_res_blocks=
                int(
                    teacher_model_cfg.get(
                        "num_res_blocks",
                        9,
                    )
                ),
            padding_type=
                teacher_model_cfg.get(
                    "padding_type",
                    "reflect",
                ),
        ).to(
            self.device
        )

        self.teacher.load_state_dict(
            teacher_checkpoint_payload[
                "model_state_dict"
            ]
        )

        self.teacher.eval()

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.teacher_best_fid = float(
            teacher_checkpoint_payload.get(
                "best_fid",
                np.nan,
            )
        )

        self.teacher_epoch = int(
            teacher_checkpoint_payload.get(
                "epoch",
                -1,
            )
        )

        print(
            "Teacher checkpoint epoch:",
            self.teacher_epoch,
        )

        print(
            "Teacher best FID:",
            self.teacher_best_fid,
        )

        print(
            "Teacher parameters trainable:",
            sum(
                int(parameter.requires_grad)
                for parameter
                in self.teacher.parameters()
            ),
        )

        # -------------------------------------------------------------
        # Representation hooks
        # -------------------------------------------------------------
        student_feature_name, student_feature_module = (
            _find_distillation_module(
                self.model.G_XY
            )
        )

        teacher_feature_name, teacher_feature_module = (
            _find_distillation_module(
                self.teacher.G_XY
            )
        )

        self.student_feature_capture = _FeatureCapture(
            student_feature_module,
            student_feature_name,
        )

        self.teacher_feature_capture = _FeatureCapture(
            teacher_feature_module,
            teacher_feature_name,
        )

        print(
            "Student feature layer:",
            student_feature_name,
        )

        print(
            "Teacher feature layer:",
            teacher_feature_name,
        )

        print(
            "KD prediction weight:",
            self.lambda_kd_prediction,
        )

        print(
            "KD feature weight:",
            self.lambda_kd_feature,
        )

        print(
            "Student inference input:",
            "single DAPI-filter AF / C2 only",
        )

        print("=" * 72)
        print()

    # =============================================================
    # Input conversion
    # =============================================================

    def _student_x_from_maf(
        self,
        maf_x,
    ):
        """
        Convert the teacher's paired MAF input [C0, C2] to the
        student's single-band DAPI-filter input [C2].
        """

        if maf_x.ndim != 4:
            raise ValueError(
                f"Expected MAF tensor [B,C,H,W], got {maf_x.shape}."
            )

        if maf_x.shape[1] != 2:
            raise ValueError(
                "CrossWave expects the MAF loader to return exactly "
                f"2 informative channels [C0, C2], got "
                f"{maf_x.shape[1]} channels."
            )

        index = self.dapi_channel_index

        return maf_x[
            :,
            index:index + 1,
            :,
            :,
        ]

    # =============================================================
    # CrossWave generator step
    # =============================================================

    def generator_step(
        self,
        student_x,
        teacher_x,
        real_y,
    ):
        """
        Student E-CycleGAN objective + CrossWave knowledge transfer.

        Teacher is frozen. Only its forward MAF->H&E generator is used
        for distillation.
        """

        self.optimizer_G.zero_grad(
            set_to_none=True
        )

        # ---------------------------------------------------------
        # Student forward translation and representation
        # ---------------------------------------------------------
        self.student_feature_capture.clear()

        fake_y = self.model.G_XY(
            student_x
        )

        student_feature = (
            self.student_feature_capture.output
        )

        if student_feature is None:
            raise RuntimeError(
                "Student feature hook did not capture a tensor."
            )

        # Keep a reference to the FIRST G_XY pass. The hook will be
        # overwritten later by reconstructed_y.
        student_feature_for_kd = (
            student_feature
        )

        # ---------------------------------------------------------
        # Frozen teacher guidance from paired full MAF
        # ---------------------------------------------------------
        self.teacher_feature_capture.clear()

        with torch.no_grad():
            teacher_fake_y = (
                self.teacher.G_XY(
                    teacher_x
                )
            )

            teacher_feature = (
                self.teacher_feature_capture.output
            )

            if teacher_feature is None:
                raise RuntimeError(
                    "Teacher feature hook did not capture a tensor."
                )

            teacher_feature_for_kd = (
                teacher_feature.detach()
            )

            teacher_fake_y = (
                teacher_fake_y.detach()
            )

        if (
            student_feature_for_kd.shape
            != teacher_feature_for_kd.shape
        ):
            raise RuntimeError(
                "Teacher/student feature shapes do not match: "
                f"student={tuple(student_feature_for_kd.shape)}, "
                f"teacher={tuple(teacher_feature_for_kd.shape)}. "
                "Use the same generator architecture/depth for both."
            )

        if (
            fake_y.shape
            != teacher_fake_y.shape
        ):
            raise RuntimeError(
                "Teacher/student output shapes do not match: "
                f"student={tuple(fake_y.shape)}, "
                f"teacher={tuple(teacher_fake_y.shape)}."
            )

        # ---------------------------------------------------------
        # Remaining student E-CycleGAN paths
        # ---------------------------------------------------------
        fake_x = self.model.G_YX(
            real_y
        )

        reconstructed_x = (
            self.model.G_YX(
                fake_y
            )
        )

        reconstructed_y = (
            self.model.G_XY(
                fake_x
            )
        )

        pred_fake_y = self.model.D_Y(
            fake_y
        )

        pred_fake_x = self.model.D_X(
            fake_x
        )

        # ---------------------------------------------------------
        # Ordinary student E-CycleGAN objective
        # ---------------------------------------------------------
        base_loss_dict = (
            self.criterion.generator_loss(
                real_x=
                    student_x,
                real_y=
                    real_y,
                fake_x=
                    fake_x,
                fake_y=
                    fake_y,
                reconstructed_x=
                    reconstructed_x,
                reconstructed_y=
                    reconstructed_y,
                pred_fake_x=
                    pred_fake_x,
                pred_fake_y=
                    pred_fake_y,
            )
        )

        base_loss = base_loss_dict[
            "total"
        ]

        # ---------------------------------------------------------
        # CrossWave prediction distillation
        #
        # Both teacher and student outputs are in the SAME inverted
        # H&E model domain, so direct L1 matching is valid.
        # ---------------------------------------------------------
        loss_kd_prediction = F.l1_loss(
            fake_y,
            teacher_fake_y,
            reduction="mean",
        )

        # ---------------------------------------------------------
        # CrossWave feature distillation
        #
        # Directly calibrates the single-band student representation
        # to the corresponding multispectral teacher representation.
        # No dimensionality reduction / adapter is used because the
        # teacher and student have the same residual representation
        # dimensionality.
        # ---------------------------------------------------------
        loss_kd_feature = F.l1_loss(
            student_feature_for_kd,
            teacher_feature_for_kd,
            reduction="mean",
        )

        total_loss = (
            base_loss
            + self.lambda_kd_prediction
            * loss_kd_prediction
            + self.lambda_kd_feature
            * loss_kd_feature
        )

        total_loss.backward()

        self.optimizer_G.step()

        return {
            "total":
                float(
                    total_loss.detach().item()
                ),
            "base":
                float(
                    base_loss.detach().item()
                ),
            "cycle":
                float(
                    base_loss_dict[
                        "cycle"
                    ]
                    .detach()
                    .item()
                ),
            "feature":
                float(
                    base_loss_dict[
                        "feature"
                    ]
                    .detach()
                    .item()
                ),
            "saliency":
                float(
                    base_loss_dict[
                        "saliency"
                    ]
                    .detach()
                    .item()
                ),
            "kd_prediction":
                float(
                    loss_kd_prediction
                    .detach()
                    .item()
                ),
            "kd_feature":
                float(
                    loss_kd_feature
                    .detach()
                    .item()
                ),
            "fake_x":
                fake_x.detach(),
            "fake_y":
                fake_y.detach(),
        }

    # =============================================================
    # Training epoch
    # =============================================================

    def train_epoch(
        self,
        epoch,
    ):
        self.model.train()
        self.teacher.eval()

        start_time = time.time()

        running = {
            "G": 0.0,
            "base_G": 0.0,
            "D_X": 0.0,
            "D_Y": 0.0,
            "cycle": 0.0,
            "feature": 0.0,
            "saliency": 0.0,
            "kd_prediction": 0.0,
            "kd_feature": 0.0,
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
            # The ordinary MAF dataset supplies the perfectly paired
            # teacher input [C0,C2].
            teacher_x = batch[
                "x"
            ].to(
                self.device,
                non_blocking=True,
            )

            # Student sees ONLY C2 / DAPI-filter AF.
            student_x = (
                self._student_x_from_maf(
                    teacher_x
                )
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

            result = self.generator_step(
                student_x=
                    student_x,
                teacher_x=
                    teacher_x,
                real_y=
                    real_y,
            )

            pooled_fake_x = (
                self.fake_x_pool.query(
                    result[
                        "fake_x"
                    ]
                )
            )

            pooled_fake_y = (
                self.fake_y_pool.query(
                    result[
                        "fake_y"
                    ]
                )
            )

            loss_D_X = (
                self.discriminator_x_step(
                    student_x,
                    pooled_fake_x,
                )
            )

            loss_D_Y = (
                self.discriminator_y_step(
                    real_y,
                    pooled_fake_y,
                )
            )

            running["G"] += result["total"]
            running["base_G"] += result["base"]
            running["D_X"] += loss_D_X
            running["D_Y"] += loss_D_Y
            running["cycle"] += result["cycle"]
            running["feature"] += result["feature"]
            running["saliency"] += result["saliency"]
            running["kd_prediction"] += result["kd_prediction"]
            running["kd_feature"] += result["kd_feature"]

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
                    f"G {result['total']:.4f} | "
                    f"base {result['base']:.4f} | "
                    f"KDpred {result['kd_prediction']:.4f} | "
                    f"KDfeat {result['kd_feature']:.4f} | "
                    f"D_X {loss_D_X:.4f} | "
                    f"D_Y {loss_D_Y:.4f} | "
                    f"cyc {result['cycle']:.4f} | "
                    f"{elapsed:.1f}s"
                )

        denominator = max(
            total_batches,
            1,
        )

        for key in running:
            running[key] /= (
                denominator
            )

        return running

    # =============================================================
    # Deterministic validation
    # =============================================================

    @torch.no_grad()
    def validate(
        self,
    ):
        self.model.eval()
        self.teacher.eval()

        # ---------------------------------------------------------
        # Paired MAF -> student C2 validation
        # ---------------------------------------------------------
        x_total_loss = 0.0
        x_total_count = 0

        kd_pred_total = 0.0
        kd_feat_total = 0.0

        for batch in self.val_x_loader:
            teacher_x = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
            )

            student_x = (
                self._student_x_from_maf(
                    teacher_x
                )
            )

            self.student_feature_capture.clear()

            fake_y = (
                self.model.G_XY(
                    student_x
                )
            )

            student_feature = (
                self.student_feature_capture.output
            )

            reconstructed_x = (
                self.model.G_YX(
                    fake_y
                )
            )

            cycle_loss = F.l1_loss(
                reconstructed_x,
                student_x,
                reduction="mean",
            )

            self.teacher_feature_capture.clear()

            teacher_fake_y = (
                self.teacher.G_XY(
                    teacher_x
                )
            )

            teacher_feature = (
                self.teacher_feature_capture.output
            )

            kd_pred = F.l1_loss(
                fake_y,
                teacher_fake_y,
                reduction="mean",
            )

            kd_feat = F.l1_loss(
                student_feature,
                teacher_feature,
                reduction="mean",
            )

            batch_size = int(
                student_x.shape[0]
            )

            x_total_loss += (
                float(
                    cycle_loss.item()
                )
                * batch_size
            )

            kd_pred_total += (
                float(
                    kd_pred.item()
                )
                * batch_size
            )

            kd_feat_total += (
                float(
                    kd_feat.item()
                )
                * batch_size
            )

            x_total_count += (
                batch_size
            )

        val_cycle_x = (
            x_total_loss
            / max(
                x_total_count,
                1,
            )
        )

        val_kd_prediction = (
            kd_pred_total
            / max(
                x_total_count,
                1,
            )
        )

        val_kd_feature = (
            kd_feat_total
            / max(
                x_total_count,
                1,
            )
        )

        # ---------------------------------------------------------
        # Y -> student X -> Y
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

            loss = F.l1_loss(
                reconstructed_y,
                real_y,
                reduction="mean",
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

            y_total_count += (
                batch_size
            )

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
            "val_kd_prediction":
                val_kd_prediction,
            "val_kd_feature":
                val_kd_feature,
        }

    # =============================================================
    # FID / KID: student uses only C2
    # =============================================================

    @torch.no_grad()
    def evaluate_fid_kid(
        self,
    ):
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
            "Evaluating CrossWave student FID/KID..."
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

            student_x = (
                self._student_x_from_maf(
                    maf_x
                )
            )

            fake_y = (
                self.model.G_XY(
                    student_x
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
                or (batch_idx + 1) % 50 == 0
                or (batch_idx + 1) == total_batches
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

        # Base trainer was configured with
        # reset_real_features=False, so reset clears fake features
        # while preserving cached real H&E statistics/features.
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
        Save student previews in physical H&E space.

        The student preview receives only C2.  A second teacher-reference
        preview is also saved from the exact same validation MAF samples.
        """

        self.model.eval()
        self.teacher.eval()

        preview_dir = (
            self.preview_dir
            / f"epoch_{epoch:03d}"
        )

        # ---------------------------------------------------------
        # Prepare single-band student X batch
        # ---------------------------------------------------------
        student_x_batch = copy.copy(
            self.fixed_val_x
        )

        student_x_batch[
            "image"
        ] = self._student_x_from_maf(
            self.fixed_val_x[
                "image"
            ]
        )

        trainer = self

        class _StudentPhysicalView(
            torch.nn.Module
        ):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def G_XY(self, x):
                return (
                    trainer._restore_y_from_model(
                        self.model.G_XY(
                            x
                        )
                    )
                )

            def G_YX(self, y_physical):
                return (
                    self.model.G_YX(
                        trainer._prepare_y_for_model(
                            y_physical
                        )
                    )
                )

        class _TeacherPhysicalView(
            torch.nn.Module
        ):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def G_XY(self, x):
                return (
                    trainer._restore_y_from_model(
                        self.model.G_XY(
                            x
                        )
                    )
                )

            def G_YX(self, y_physical):
                return (
                    self.model.G_YX(
                        trainer._prepare_y_for_model(
                            y_physical
                        )
                    )
                )

        save_translation_preview(
            model=
                _StudentPhysicalView(
                    self.model
                ),
            x_batch=
                student_x_batch,
            y_batch=
                self.fixed_val_y,
            output_dir=
                preview_dir
                / "student",
            device=
                self.device,
            max_samples=4,
        )

        save_translation_preview(
            model=
                _TeacherPhysicalView(
                    self.teacher
                ),
            x_batch=
                self.fixed_val_x,
            y_batch=
                self.fixed_val_y,
            output_dir=
                preview_dir
                / "teacher_reference",
            device=
                self.device,
            max_samples=4,
        )

    # =============================================================
    # History additions
    # =============================================================

    def save_history_row(
        self,
        epoch,
        train_metrics,
        val_metrics,
        distribution_metrics,
        lrs,
        next_lrs,
    ):
        # Let the base trainer write all standard E-CycleGAN,
        # FID/KID, and scheduler columns first.
        super().save_history_row(
            epoch=
                epoch,
            train_metrics=
                train_metrics,
            val_metrics=
                val_metrics,
            distribution_metrics=
                distribution_metrics,
            lrs=
                lrs,
            next_lrs=
                next_lrs,
        )

        df = pd.read_csv(
            self.history_path
        )

        mask = (
            df[
                "epoch"
            ]
            == epoch
        )

        df.loc[
            mask,
            "train_base_G",
        ] = train_metrics[
            "base_G"
        ]

        df.loc[
            mask,
            "train_kd_prediction",
        ] = train_metrics[
            "kd_prediction"
        ]

        df.loc[
            mask,
            "train_kd_feature",
        ] = train_metrics[
            "kd_feature"
        ]

        df.loc[
            mask,
            "val_kd_prediction",
        ] = val_metrics[
            "val_kd_prediction"
        ]

        df.loc[
            mask,
            "val_kd_feature",
        ] = val_metrics[
            "val_kd_feature"
        ]

        df.loc[
            mask,
            "teacher_epoch",
        ] = self.teacher_epoch

        df.loc[
            mask,
            "teacher_best_fid",
        ] = self.teacher_best_fid

        df.loc[
            mask,
            "lambda_kd_prediction",
        ] = self.lambda_kd_prediction

        df.loc[
            mask,
            "lambda_kd_feature",
        ] = self.lambda_kd_feature

        df.to_csv(
            self.history_path,
            index=False,
        )

    # =============================================================
    # Checkpoint additions
    # =============================================================

    def _checkpoint_payload(
        self,
        epoch,
    ):
        payload = super()._checkpoint_payload(
            epoch
        )

        payload[
            "crosswave"
        ] = {
            "teacher_checkpoint":
                str(
                    self.teacher_checkpoint
                ),
            "teacher_epoch":
                self.teacher_epoch,
            "teacher_best_fid":
                self.teacher_best_fid,
            "dapi_channel_index":
                self.dapi_channel_index,
            "lambda_kd_prediction":
                self.lambda_kd_prediction,
            "lambda_kd_feature":
                self.lambda_kd_feature,
            "feature_layer_student":
                self.student_feature_capture.name,
            "feature_layer_teacher":
                self.teacher_feature_capture.name,
        }

        return payload

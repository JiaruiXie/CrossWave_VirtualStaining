from src.data.subeval_dataset import (
    SubEvalTileDataset,
)

from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import torchmetrics

from torchmetrics.image.fid import (
    FrechetInceptionDistance,
)

from torchmetrics.image.kid import (
    KernelInceptionDistance,
)


from src.data.datasets import (
    ECycleGANDataset,
    SingleDomainDataset,
)

from src.models.ecyclegan import (
    ECycleGAN,
)

from src.losses.ecyclegan_loss import (
    ECycleGANLoss,
)

from src.utils.visualization import (
    save_translation_preview,
)

class ImagePool:
    """
    CycleGAN-style fake-image replay buffer.

    Newly generated images are mixed with previously generated
    images when training the discriminator.

    pool_size=0 disables replay.
    """

    def __init__(
        self,
        pool_size=50,
    ):
        self.pool_size = int(
            pool_size
        )

        self.images = []

    @torch.no_grad()
    def query(
        self,
        images,
    ):
        """
        Parameters
        ----------
        images:
            Tensor [B, C, H, W].

        Returns
        -------
        Tensor
            Batch of current and/or historical fake images.
        """

        if self.pool_size <= 0:
            return images.detach()

        returned_images = []

        for image in images.detach():

            image = image.unsqueeze(
                0
            )

            # ---------------------------------------------
            # Fill pool initially
            # ---------------------------------------------
            if len(
                self.images
            ) < self.pool_size:

                self.images.append(
                    image.clone()
                )

                returned_images.append(
                    image
                )

            else:

                # -----------------------------------------
                # Standard CycleGAN behavior:
                #
                # 50% probability:
                # return an old fake and replace it
                #
                # otherwise:
                # use current fake
                # -----------------------------------------

                if np.random.rand() > 0.5:

                    index = np.random.randint(
                        0,
                        self.pool_size,
                    )

                    old_image = (
                        self.images[
                            index
                        ]
                        .clone()
                    )

                    self.images[
                        index
                    ] = image.clone()

                    returned_images.append(
                        old_image
                    )

                else:

                    returned_images.append(
                        image
                    )

        return torch.cat(
            returned_images,
            dim=0,
        )

# =========================================================
# E-CycleGAN Trainer
# =========================================================

class ECycleGANTrainer:

    def __init__(
        self,
        config,
        root,
    ):
        """
        E-CycleGAN trainer.

        Supports:

        - MAF teacher:
              X = 2-channel MAF [C0, C2]
              Y = 3-channel H&E

        - DAPI baseline:
              X = 1-channel DAPI-filter AF
              Y = 3-channel H&E

        Includes:

        - training
        - deterministic cycle validation
        - per-epoch FID
        - per-epoch KID
        - fixed preview generation
        - checkpointing
        - best val-cycle checkpoint
        - best FID checkpoint
        """

        self.config = config

        self.project_root = Path(
            root
        )

        # =================================================
        # Config sections
        # =================================================

        self.data_cfg = config[
            "data"
        ]

        self.model_cfg = config[
            "model"
        ]

        self.training_cfg = config[
            "training"
        ]

        self.loss_cfg = config[
            "loss"
        ]

        self.checkpoint_cfg = config.get(
            "checkpoint",
            {},
        )

        self.logging_cfg = config.get(
            "logging",
            {},
        )

        self.experiment_name = config.get(
            "experiment",
            "ecyclegan",
        )

        self.seed = int(
            config.get(
                "seed",
                42,
            )
        )

        self.start_epoch = 1

        # =================================================
        # Seed
        # =================================================

        self._set_seed(
            self.seed
        )

        # =================================================
        # Device
        # =================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Training device:",
            self.device,
        )

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(
                    0
                ),
            )

        # =========================================================
        # Experiment / output directories
        # =========================================================

        experiment_cfg = config.get(
            "experiment",
            {}
        )

        # Support both:
        #
        # experiment: maf_teacher
        #
        # and:
        #
        # experiment:
        #   name: maf_teacher

        if isinstance(
            experiment_cfg,
            dict,
        ):
            self.experiment_name = experiment_cfg.get(
                "name",
                "maf_teacher",
            )
        else:
            self.experiment_name = str(
                experiment_cfg
            )

        print(
            "Experiment name:",
            self.experiment_name,
        )

        self.output_dir = (
            self.project_root
            / "outputs"
            / self.experiment_name
        )

        self.checkpoint_dir = (
            self.output_dir
            / "checkpoints"
        )

        self.preview_dir = (
            self.output_dir
            / "previews"
        )

        self.evaluation_dir = (
            self.output_dir
            / "evaluation"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.preview_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.evaluation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history_path = (
            self.output_dir
            / "training_history.csv"
        )

        # =================================================
        # Reusable real-H&E Inception cache
        #
        # The Sub_eval real H&E distribution is identical
        # across experiments, so its FID/KID Inception
        # features do not need to be recomputed every time.
        #
        # Default:
        #
        # data/cache/
        #   sub_eval_real_he_inception_fid_kid.pt
        #
        # The path can be overridden in YAML with:
        #
        # training:
        #   real_metric_cache:
        #     data/cache/your_cache_name.pt
        #
        # Set:
        #
        # training:
        #   reuse_real_metric_cache: false
        #
        # to force recomputation without deleting the file.
        # =================================================

        metric_cache_path = Path(
            self.training_cfg.get(
                "real_metric_cache",
                (
                    "data/cache/"
                    "sub_eval_real_he_inception_fid_kid.pt"
                ),
            )
        )

        if not metric_cache_path.is_absolute():

            metric_cache_path = (
                self.project_root
                / metric_cache_path
            )

        self.real_metric_cache_path = (
            metric_cache_path
        )

        self.real_metric_cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.reuse_real_metric_cache = bool(
            self.training_cfg.get(
                "reuse_real_metric_cache",
                True,
            )
        )

        # =================================================
        # Dataset mode
        # =================================================

        self.mode = self.data_cfg.get(
            "mode",
            "maf",
        )

        # =================================================
        # Y-domain inversion
        #
        # If enabled, the model is trained in an inverted
        # H&E domain:
        #
        #     y_model = -y_physical
        #
        # This transformation is applied consistently to
        # all Y-domain tensors used by the model:
        #
        #   - real Y input to G_YX
        #   - real Y input to D_Y
        #   - Y cycle-consistency target
        #   - Y validation
        #
        # G_XY therefore learns to generate inverted H&E.
        #
        # For physical H&E visualization and FID/KID,
        # generated Y is transformed back with:
        #
        #     y_physical = -y_model
        # =================================================

        self.invert_y_domain = bool(
            self.data_cfg.get(
                "invert_y_domain",
                False,
            )
        )

        print(
            "Invert Y domain:",
            self.invert_y_domain,
        )

        # =================================================
        # Split CSV
        # =================================================

        split_csv = Path(
            self.data_cfg[
                "split_csv"
            ]
        )

        if not split_csv.is_absolute():

            split_csv = (
                self.project_root
                / split_csv
            )

        self.split_csv = split_csv

        # =================================================
        # Training dataset
        # =================================================

        self.train_dataset = (
            ECycleGANDataset(
                split_csv=
                    self.split_csv,

                split="train",

                mode=
                    self.mode,

                augment=
                    bool(
                        self.data_cfg.get(
                            "augment",
                            True,
                        )
                    ),
            )
        )

        # =================================================
        # Validation datasets
        #
        # X and Y deliberately validated separately.
        # Every validation image is evaluated once.
        # =================================================

        self.val_x_dataset = (
            SingleDomainDataset(
                split_csv=
                    self.split_csv,

                split="val",

                domain="x",

                mode=
                    self.mode,
            )
        )

        self.val_y_dataset = (
            SingleDomainDataset(
                split_csv=
                    self.split_csv,

                split="val",

                domain="y",

                mode=
                    self.mode,
            )
        )

        # =================================================
        # DataLoaders
        # =================================================

        batch_size = int(
            self.data_cfg.get(
                "batch_size",
                4,
            )
        )

        num_workers = int(
            self.data_cfg.get(
                "num_workers",
                4,
            )
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=
                batch_size,
            shuffle=True,
            num_workers=
                num_workers,
            pin_memory=
                torch.cuda.is_available(),
            drop_last=False,
        )

        # Cycle validation can use a larger batch.
        val_batch_size = int(
            self.data_cfg.get(
                "val_batch_size",
                batch_size,
            )
        )

        self.val_x_loader = DataLoader(
            self.val_x_dataset,
            batch_size=
                val_batch_size,
            shuffle=False,
            num_workers=
                num_workers,
            pin_memory=
                torch.cuda.is_available(),
            drop_last=False,
        )

        self.val_y_loader = DataLoader(
            self.val_y_dataset,
            batch_size=
                val_batch_size,
            shuffle=False,
            num_workers=
                num_workers,
            pin_memory=
                torch.cuda.is_available(),
            drop_last=False,
        )

        # =================================================
        # Fixed preview batches
        # =================================================

        preview_x_loader = DataLoader(
            self.val_x_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        preview_y_loader = DataLoader(
            self.val_y_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        self.fixed_val_x = next(
            iter(
                preview_x_loader
            )
        )

        self.fixed_val_y = next(
            iter(
                preview_y_loader
            )
        )

        # =================================================
        # Model
        # =================================================

        self.model = ECycleGAN(

            x_channels=
                self.model_cfg[
                    "x_channels"
                ],

            y_channels=
                self.model_cfg[
                    "y_channels"
                ],

            base_channels=
                self.model_cfg.get(
                    "base_channels",
                    64,
                ),

            num_res_blocks=
                self.model_cfg.get(
                    "num_res_blocks",
                    9,
                ),

            padding_type=
                self.model_cfg.get(
                    "padding_type",
                    "reflect",
                ),

        ).to(
            self.device
        )

        # -------------------------------------------------
        # Explicit channel sanity check
        # -------------------------------------------------

        configured_x_channels = int(
            self.model_cfg[
                "x_channels"
            ]
        )

        actual_x_channels = (
            self.model
            .G_XY
            .initial[1]
            .in_channels
        )

        print(
            "Configured X channels:",
            configured_x_channels,
        )

        print(
            "G_XY first conv in_channels:",
            actual_x_channels,
        )

        if (
            configured_x_channels
            != actual_x_channels
        ):

            raise RuntimeError(
                "Model input-channel mismatch: "
                f"config={configured_x_channels}, "
                f"model={actual_x_channels}"
            )

        # =================================================
        # Loss
        # =================================================

        self.criterion = ECycleGANLoss(

            lambda_cyc=
                float(
                    self.loss_cfg.get(
                        "lambda_cyc",
                        10.0,
                    )
                ),

            lambda_feat=
                float(
                    self.loss_cfg.get(
                        "lambda_feat",
                        1.0,
                    )
                ),

            lambda_sal=
                float(
                    self.loss_cfg.get(
                        "lambda_sal",
                        1.0,
                    )
                ),

            saliency_threshold=
                float(
                    self.loss_cfg.get(
                        "threshold",
                        170,
                    )
                ),

            saliency_sharpness=
                float(
                    self.loss_cfg.get(
                        "sharpness",
                        20.0,
                    )
                ),

            auxiliary_invert_x=
                bool(
                    self.loss_cfg.get(
                        "auxiliary_invert_x",
                        False,
                    )
                ),

        ).to(
            self.device
        )

        print(
            "Auxiliary X inversion:",
            self.criterion.auxiliary_invert_x,
        )

        if (
            self.invert_y_domain
            and self.criterion.auxiliary_invert_x
        ):
            raise ValueError(
                "For the inverted-Y experiment, set "
                "loss.auxiliary_invert_x: false. "
                "The purpose of this experiment is to align "
                "raw AF with the inverted H&E target domain "
                "without also inverting X inside the auxiliary "
                "losses."
            )

        # =========================================================
        # Optimizer learning rates
        # =========================================================

        # Backward-compatible fallback:
        # if generator_lr/discriminator_lr are absent,
        # use the original "lr" value.

        default_lr = float(
            self.training_cfg.get(
                "lr",
                0.0002,
            )
        )

        generator_lr = float(
            self.training_cfg.get(
                "generator_lr",
                default_lr,
            )
        )

        discriminator_lr = float(
            self.training_cfg.get(
                "discriminator_lr",
                default_lr,
            )
        )

        beta1 = float(
            self.training_cfg.get(
                "beta1",
                0.5,
            )
        )

        beta2 = float(
            self.training_cfg.get(
                "beta2",
                0.999,
            )
        )

        print(
            f"Generator LR:     {generator_lr:.8f}"
        )

        print(
            f"Discriminator LR: {discriminator_lr:.8f}"
        )

        generator_parameters = list(
            self.model.G_XY.parameters()
        ) + list(
            self.model.G_YX.parameters()
        )

        self.optimizer_G = torch.optim.Adam(
            generator_parameters,
            lr=generator_lr,
            betas=(
                beta1,
                beta2,
            ),
        )

        self.optimizer_D_X = torch.optim.Adam(
            self.model.D_X.parameters(),
            lr=discriminator_lr,
            betas=(
                beta1,
                beta2,
            ),
        )

        self.optimizer_D_Y = torch.optim.Adam(
            self.model.D_Y.parameters(),
            lr=discriminator_lr,
            betas=(
                beta1,
                beta2,
            ),
        )

        # =========================================================
        # Fake-image replay buffers
        # =========================================================

        fake_pool_size = int(
            self.training_cfg.get(
                "fake_pool_size",
                0,
            )
        )

        self.fake_x_pool = ImagePool(
            pool_size=fake_pool_size
        )

        self.fake_y_pool = ImagePool(
            pool_size=fake_pool_size
        )

        print(
            "Fake-image pool size:",
            fake_pool_size,
        )

        # =================================================
        # LR scheduler: FID-driven ReduceLROnPlateau
        #
        # Motivation:
        # The teacher reached a useful adversarial equilibrium
        # very early, while the discriminators continued to
        # strengthen under the original 100-epoch fixed-LR
        # schedule.  Here we reduce learning rates when Sub_eval
        # FID stops improving.  Discriminator LRs decay more
        # aggressively than the generator LR.
        # =================================================

        self.scheduler_type = str(
            self.training_cfg.get(
                "lr_scheduler",
                "reduce_on_plateau",
            )
        ).lower()

        self.plateau_factor_g = float(
            self.training_cfg.get(
                "plateau_factor_g",
                0.5,
            )
        )

        self.plateau_factor_d = float(
            self.training_cfg.get(
                "plateau_factor_d",
                0.25,
            )
        )

        self.plateau_patience = int(
            self.training_cfg.get(
                "plateau_patience",
                2,
            )
        )

        self.plateau_threshold = float(
            self.training_cfg.get(
                "plateau_threshold",
                0.5,
            )
        )

        self.plateau_min_lr_g = float(
            self.training_cfg.get(
                "plateau_min_lr_g",
                1e-6,
            )
        )

        self.plateau_min_lr_d = float(
            self.training_cfg.get(
                "plateau_min_lr_d",
                1e-6,
            )
        )

        if self.scheduler_type not in {
            "reduce_on_plateau",
            "plateau",
            "fid_plateau",
        }:
            raise ValueError(
                "This trainer expects an FID-driven "
                "ReduceLROnPlateau scheduler. "
                f"Received lr_scheduler={self.scheduler_type!r}."
            )

        self.scheduler_type = "reduce_on_plateau"

        self.scheduler_G = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_G,
                mode="min",
                factor=self.plateau_factor_g,
                patience=self.plateau_patience,
                threshold=self.plateau_threshold,
                threshold_mode="abs",
                cooldown=0,
                min_lr=self.plateau_min_lr_g,
            )
        )

        self.scheduler_D_X = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_D_X,
                mode="min",
                factor=self.plateau_factor_d,
                patience=self.plateau_patience,
                threshold=self.plateau_threshold,
                threshold_mode="abs",
                cooldown=0,
                min_lr=self.plateau_min_lr_d,
            )
        )

        self.scheduler_D_Y = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_D_Y,
                mode="min",
                factor=self.plateau_factor_d,
                patience=self.plateau_patience,
                threshold=self.plateau_threshold,
                threshold_mode="abs",
                cooldown=0,
                min_lr=self.plateau_min_lr_d,
            )
        )

        # =================================================
        # Early stopping on FID
        # =================================================

        self.early_stopping_patience = int(
            self.training_cfg.get(
                "early_stopping_patience",
                8,
            )
        )

        self.early_stopping_min_delta = float(
            self.training_cfg.get(
                "early_stopping_min_delta",
                0.0,
            )
        )

        self.epochs_without_fid_improvement = 0

        print(
            "LR scheduler:",
            "ReduceLROnPlateau on Sub_eval FID",
        )

        print(
            "  Generator factor:",
            self.plateau_factor_g,
        )

        print(
            "  Discriminator factor:",
            self.plateau_factor_d,
        )

        print(
            "  Patience:",
            self.plateau_patience,
        )

        print(
            "  FID threshold:",
            self.plateau_threshold,
        )

        print(
            "Early stopping patience:",
            self.early_stopping_patience,
            "FID evaluations",
        )

        # =================================================
        # Best cycle metric
        # =================================================

        self.best_val_cycle_x = float(
            "inf"
        )

        # =================================================
        # FID / KID settings
        # =================================================

        self.fid_enabled = bool(
            self.training_cfg.get(
                "compute_fid_kid",
                True,
            )
        )

        self.fid_every = int(
            self.training_cfg.get(
                "fid_every",
                1,
            )
        )

        self.kid_subsets = int(
            self.training_cfg.get(
                "kid_subsets",
                50,
            )
        )

        self.kid_subset_size = int(
            self.training_cfg.get(
                "kid_subset_size",
                1000,
            )
        )

        metric_batch_size = int(
            self.training_cfg.get(
                "metric_batch_size",
                16,
            )
        )

        # =========================================================
        # Sub_eval FID / KID datasets
        # =========================================================

        sub_eval_root = (
            self.project_root
            / "data"
            / "raw"
            / "Sub_eval"
        )

        sub_eval_tile_size = int(
            self.training_cfg.get(
                "sub_eval_tile_size",
                256,
            )
        )

        sub_eval_stride = int(
            self.training_cfg.get(
                "sub_eval_stride",
                256,
            )
        )

        self.metric_x_dataset = (
            SubEvalTileDataset(
                root=
                    sub_eval_root,

                domain="af",

                tile_size=
                    sub_eval_tile_size,

                stride=
                    sub_eval_stride,
            )
        )

        self.metric_y_dataset = (
            SubEvalTileDataset(
                root=
                    sub_eval_root,

                domain="he",

                tile_size=
                    sub_eval_tile_size,

                stride=
                    sub_eval_stride,
            )
        )

        metric_batch_size = int(
            self.training_cfg.get(
                "metric_batch_size",
                16,
            )
        )

        self.metric_x_loader = DataLoader(
            self.metric_x_dataset,
            batch_size=
                metric_batch_size,
            shuffle=False,
            num_workers=
                num_workers,
            pin_memory=
                torch.cuda.is_available(),
            drop_last=False,
        )

        self.metric_y_loader = DataLoader(
            self.metric_y_dataset,
            batch_size=
                metric_batch_size,
            shuffle=False,
            num_workers=
                num_workers,
            pin_memory=
                torch.cuda.is_available(),
            drop_last=False,
        )

        print()
        print(
            "Sub_eval validation:"
        )

        print(
            "  AF tiles:",
            len(
                self.metric_x_dataset
            ),
        )

        print(
            "  Real H&E tiles:",
            len(
                self.metric_y_dataset
            ),
        )

        # =================================================
        # Historical best FID
        # =================================================

        self.best_fid = (
            self._recover_historical_best_fid()
        )

        print(
            "Historical best FID:",
            (
                f"{self.best_fid:.6f}"
                if np.isfinite(
                    self.best_fid
                )
                else "None"
            ),
        )

        self.best_fid_epoch = (
            self._recover_historical_best_fid_epoch()
        )

        if (
            self.best_fid_epoch
            is not None
        ):

            print(
                "Historical best FID epoch:",
                self.best_fid_epoch,
            )

        # =================================================
        # FID / KID objects
        # =================================================

        self.fid_metric = None
        self.kid_metric = None

        if self.fid_enabled:

            print()
            print(
                "Initializing FID/KID metrics..."
            )

            self.fid_metric = (
                FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                    reset_real_features=False,
                )
                .set_dtype(
                    torch.float64
                )
                .to(
                    self.device
                )
            )

            self.kid_metric = (
                KernelInceptionDistance(
                    feature=2048,
                    subsets=
                        self.kid_subsets,
                    subset_size=
                        self.kid_subset_size,
                    normalize=True,
                    reset_real_features=False,
                )
                .to(
                    self.device
                )
            )

            # ---------------------------------------------
            # TorchMetrics metric states are non-persistent
            # by default. Make them persistent so the
            # accumulated real-image state appears in
            # state_dict() and can be saved to disk.
            # ---------------------------------------------

            self.fid_metric.persistent(
                True
            )

            self.kid_metric.persistent(
                True
            )

            # ---------------------------------------------
            # Load reusable real-H&E Inception state when
            # possible. Otherwise compute it once and save.
            # ---------------------------------------------

            cache_loaded = False

            if self.reuse_real_metric_cache:

                cache_loaded = (
                    self._load_real_metric_cache()
                )

            else:

                print(
                    "Real metric cache reuse disabled."
                )

            if not cache_loaded:

                self._initialize_real_metric_features()

                self._save_real_metric_cache()

    # =====================================================
    # Reproducibility
    # =====================================================

    def _set_seed(
        self,
        seed,
    ):

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                seed
            )

    # =====================================================
    # Y-domain representation helpers
    # =====================================================

    def _prepare_y_for_model(
        self,
        y,
    ):
        """
        Convert physical H&E into the Y representation used
        internally by the model.

        Tensors are normalized to [-1, 1].

        If data.invert_y_domain=True:

            y_model = -y_physical
        """

        if self.invert_y_domain:
            return -y

        return y

    def _restore_y_from_model(
        self,
        y,
    ):
        """
        Convert model-domain Y back to physical H&E.

        The inversion is self-inverse:

            y_physical = -y_model
        """

        if self.invert_y_domain:
            return -y

        return y

    # =====================================================
    # Metric input conversion
    # =====================================================

    def _to_metric_image(
        self,
        x,
    ):
        """
        Convert network image:

            [-1, 1]

        to TorchMetrics image:

            [0, 1]
        """

        x = (
            x + 1.0
        ) / 2.0

        x = torch.clamp(
            x,
            0.0,
            1.0,
        )

        return x.float()

    # =====================================================
    # Historical FID recovery
    # =====================================================

    def _fid_result_paths(
        self,
    ):

        return [
            self.evaluation_dir
            / "fid_kid_results.csv",

            self.history_path,
        ]

    def _recover_historical_best_fid(
        self,
    ):

        candidates = []

        for path in self._fid_result_paths():

            if not path.exists():
                continue

            try:

                df = pd.read_csv(
                    path
                )

            except Exception:

                continue

            # Standalone evaluator uses "FID".
            if "FID" in df.columns:

                values = pd.to_numeric(
                    df["FID"],
                    errors="coerce",
                )

                values = values[
                    np.isfinite(
                        values
                    )
                ]

                if len(
                    values
                ) > 0:

                    candidates.append(
                        float(
                            values.min()
                        )
                    )

            # Trainer history uses "fid".
            if "fid" in df.columns:

                values = pd.to_numeric(
                    df["fid"],
                    errors="coerce",
                )

                values = values[
                    np.isfinite(
                        values
                    )
                ]

                if len(
                    values
                ) > 0:

                    candidates.append(
                        float(
                            values.min()
                        )
                    )

        if len(
            candidates
        ) == 0:

            return float(
                "inf"
            )

        return min(
            candidates
        )

    def _recover_historical_best_fid_epoch(
        self,
    ):

        best_rows = []

        for path in self._fid_result_paths():

            if not path.exists():
                continue

            try:

                df = pd.read_csv(
                    path
                )

            except Exception:

                continue

            if (
                "epoch"
                not in df.columns
            ):
                continue

            if "FID" in df.columns:

                tmp = df[
                    [
                        "epoch",
                        "FID",
                    ]
                ].copy()

                tmp.columns = [
                    "epoch",
                    "fid",
                ]

                best_rows.append(
                    tmp
                )

            if "fid" in df.columns:

                tmp = df[
                    [
                        "epoch",
                        "fid",
                    ]
                ].copy()

                best_rows.append(
                    tmp
                )

        if len(
            best_rows
        ) == 0:

            return None

        merged = pd.concat(
            best_rows,
            ignore_index=True,
        )

        merged["fid"] = pd.to_numeric(
            merged["fid"],
            errors="coerce",
        )

        merged["epoch"] = pd.to_numeric(
            merged["epoch"],
            errors="coerce",
        )

        merged = merged.dropna(
            subset=[
                "fid",
                "epoch",
            ]
        )

        if len(
            merged
        ) == 0:

            return None

        row = merged.loc[
            merged["fid"].idxmin()
        ]

        return int(
            row[
                "epoch"
            ]
        )

    # =====================================================
    # Reusable real-H&E FID/KID feature cache
    # =====================================================

    def _metric_cache_metadata(
        self,
    ):
        """
        Metadata used to verify that an existing cache
        matches the current Sub_eval evaluation setup.

        If any of these values change, the real features
        are recomputed automatically.
        """

        return {
            "cache_version":
                1,

            "torch_version":
                str(
                    torch.__version__
                ),

            "torchmetrics_version":
                str(
                    torchmetrics.__version__
                ),

            "fid_feature":
                2048,

            "normalize":
                True,

            "real_he_tiles":
                int(
                    len(
                        self.metric_y_dataset
                    )
                ),

            "sub_eval_tile_size":
                int(
                    self.training_cfg.get(
                        "sub_eval_tile_size",
                        256,
                    )
                ),

            "sub_eval_stride":
                int(
                    self.training_cfg.get(
                        "sub_eval_stride",
                        256,
                    )
                ),

            "sub_eval_root":
                str(
                    (
                        self.project_root
                        / "data"
                        / "raw"
                        / "Sub_eval"
                    ).resolve()
                ),
        }

    def _state_to_cpu(
        self,
        value,
    ):
        """
        Recursively move tensors in a TorchMetrics
        state_dict to CPU before writing the cache.

        This keeps the cache portable and avoids storing
        CUDA-device-specific tensors.
        """

        if torch.is_tensor(
            value
        ):

            return (
                value
                .detach()
                .cpu()
            )

        if isinstance(
            value,
            dict,
        ):

            return {
                key:
                    self._state_to_cpu(
                        item
                    )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            list,
        ):

            return [
                self._state_to_cpu(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            tuple,
        ):

            return tuple(
                self._state_to_cpu(
                    item
                )
                for item in value
            )

        return value

    def _load_real_metric_cache(
        self,
    ):
        """
        Load previously cached real-H&E FID/KID states.

        Returns
        -------
        bool
            True if a compatible cache was restored.
            False if the cache is missing, incompatible,
            or cannot be loaded.
        """

        path = (
            self.real_metric_cache_path
        )

        if not path.exists():

            print(
                "No cached real H&E metric "
                "features found."
            )

            print(
                "Expected cache:",
                path,
            )

            return False

        print()
        print(
            "Loading cached real H&E "
            "Inception features..."
        )

        print(
            "Cache:",
            path,
        )

        try:

            cache = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        except Exception as exc:

            print(
                "Warning: could not load "
                "real metric cache:"
            )

            print(
                " ",
                exc,
            )

            return False

        expected_metadata = (
            self._metric_cache_metadata()
        )

        cached_metadata = (
            cache.get(
                "metadata",
                {},
            )
            if isinstance(
                cache,
                dict,
            )
            else {}
        )

        if (
            cached_metadata
            != expected_metadata
        ):

            print(
                "Real metric cache metadata "
                "does not match current settings."
            )

            print(
                "Cached metadata:"
            )

            for key, value in (
                cached_metadata.items()
            ):

                print(
                    f"  {key}: {value}"
                )

            print(
                "Current metadata:"
            )

            for key, value in (
                expected_metadata.items()
            ):

                print(
                    f"  {key}: {value}"
                )

            print(
                "Real H&E features will be "
                "recomputed once."
            )

            return False

        if (
            "fid_metric_state"
            not in cache
            or "kid_metric_state"
            not in cache
        ):

            print(
                "Real metric cache is incomplete."
            )

            return False

        try:

            self.fid_metric.load_state_dict(
                cache[
                    "fid_metric_state"
                ],
                strict=True,
            )

            self.kid_metric.load_state_dict(
                cache[
                    "kid_metric_state"
                ],
                strict=True,
            )

        except Exception as exc:

            print(
                "Warning: cached metric states "
                "could not be restored:"
            )

            print(
                " ",
                exc,
            )

            return False

        # ---------------------------------------------
        # load_state_dict copies tensor state into the
        # metric modules. Make sure modules remain on
        # the requested training/evaluation device.
        # ---------------------------------------------

        self.fid_metric = (
            self.fid_metric.to(
                self.device
            )
        )

        self.kid_metric = (
            self.kid_metric.to(
                self.device
            )
        )

        print(
            "Cached real H&E Inception "
            "features loaded successfully."
        )

        print(
            "Skipping real H&E feature "
            "extraction."
        )

        return True

    def _save_real_metric_cache(
        self,
    ):
        """
        Save the real-H&E-only TorchMetrics states.

        This method is called immediately after real H&E
        features are initialized and before any fake
        features are added.
        """

        path = (
            self.real_metric_cache_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "metadata":
                self._metric_cache_metadata(),

            "fid_metric_state":
                self._state_to_cpu(
                    self.fid_metric.state_dict()
                ),

            "kid_metric_state":
                self._state_to_cpu(
                    self.kid_metric.state_dict()
                ),
        }

        temp_path = (
            path.parent
            / (
                path.name
                + ".tmp"
            )
        )

        try:

            torch.save(
                payload,
                temp_path,
            )

            temp_path.replace(
                path
            )

        finally:

            if temp_path.exists():

                try:

                    temp_path.unlink()

                except Exception:

                    pass

        print()
        print(
            "Real H&E Inception feature "
            "cache saved:"
        )

        print(
            " ",
            path,
        )

        try:

            size_mb = (
                path.stat().st_size
                / (
                    1024.0
                    * 1024.0
                )
            )

            print(
                f"Cache size: {size_mb:.1f} MB"
            )

        except Exception:

            pass

    # =====================================================
    # Initialize real H&E metric features
    # =====================================================

    @torch.no_grad()
    def _initialize_real_metric_features(
        self,
    ):
        """
        Real H&E validation distribution never changes.

        Extract it once and retain it in the TorchMetrics
        metric states throughout training.
        """

        print(
            "Caching real H&E Inception features..."
        )

        self.model.eval()

        total_batches = len(
            self.metric_y_loader
        )

        for batch_idx, batch in enumerate(
            self.metric_y_loader
        ):

            real_y = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
            )

            real_y_metric = (
                self._to_metric_image(
                    real_y
                )
            )

            self.fid_metric.update(
                real_y_metric,
                real=True,
            )

            self.kid_metric.update(
                real_y_metric,
                real=True,
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
                    "  Real metric batch "
                    f"{batch_idx + 1}/"
                    f"{total_batches}"
                )

        print(
            "Real H&E metric features cached."
        )

    # =====================================================
    # Generator step
    # =====================================================

    def generator_step(
        self,
        real_x,
        real_y,
    ):

        self.optimizer_G.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------
        # Forward translations
        # -------------------------------------------------

        fake_y = self.model.G_XY(
            real_x
        )

        fake_x = self.model.G_YX(
            real_y
        )

        # -------------------------------------------------
        # Cycles
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Adversarial predictions
        # -------------------------------------------------

        pred_fake_y = self.model.D_Y(
            fake_y
        )

        pred_fake_x = self.model.D_X(
            fake_x
        )

        # -------------------------------------------------
        # Combined E-CycleGAN generator loss
        # -------------------------------------------------

        loss_dict = (
            self.criterion.generator_loss(

                real_x=
                    real_x,

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

        loss_G = loss_dict[
            "total"
        ]

        loss_G.backward()

        self.optimizer_G.step()

        return (
            float(
                loss_G.detach().item()
            ),

            float(
                loss_dict[
                    "cycle"
                ]
                .detach()
                .item()
            ),

            float(
                loss_dict[
                    "feature"
                ]
                .detach()
                .item()
            ),

            float(
                loss_dict[
                    "saliency"
                ]
                .detach()
                .item()
            ),

            fake_x.detach(),

            fake_y.detach(),
        )

    # =====================================================
    # Discriminator X
    # =====================================================

    def discriminator_x_step(
        self,
        real_x,
        fake_x,
    ):

        self.optimizer_D_X.zero_grad(
            set_to_none=True
        )

        pred_real_x = self.model.D_X(
            real_x
        )

        pred_fake_x = self.model.D_X(
            fake_x.detach()
        )

        loss_dict = (
            self.criterion.discriminator_loss(
                pred_real=pred_real_x,
                pred_fake=pred_fake_x,
            )
        )

        if isinstance(loss_dict, dict):

            if "total" in loss_dict:
                loss_D_X = loss_dict["total"]

            elif "loss" in loss_dict:
                loss_D_X = loss_dict["loss"]

            else:
                raise KeyError(
                    "discriminator_loss returned a dict "
                    f"without 'total' or 'loss'. "
                    f"Keys: {list(loss_dict.keys())}"
                )

        else:
            loss_D_X = loss_dict

        loss_D_X.backward()

        self.optimizer_D_X.step()

        return float(
            loss_D_X.detach().item()
        )

    # =====================================================
    # Discriminator Y
    # =====================================================

    def discriminator_y_step(
        self,
        real_y,
        fake_y,
    ):

        self.optimizer_D_Y.zero_grad(
            set_to_none=True
        )

        pred_real_y = self.model.D_Y(
            real_y
        )

        pred_fake_y = self.model.D_Y(
            fake_y.detach()
        )

        loss_dict = (
            self.criterion.discriminator_loss(
                pred_real=pred_real_y,
                pred_fake=pred_fake_y,
            )
        )

        if isinstance(loss_dict, dict):

            if "total" in loss_dict:
                loss_D_Y = loss_dict["total"]

            elif "loss" in loss_dict:
                loss_D_Y = loss_dict["loss"]

            else:
                raise KeyError(
                    "discriminator_loss returned a dict "
                    f"without 'total' or 'loss'. "
                    f"Keys: {list(loss_dict.keys())}"
                )

        else:
            loss_D_Y = loss_dict

        loss_D_Y.backward()

        self.optimizer_D_Y.step()

        return float(
            loss_D_Y.detach().item()
        )

    # =====================================================
    # Train one epoch
    # =====================================================

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

            real_x = batch[
                "x"
            ].to(
                self.device,
                non_blocking=True,
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

            # =========================================================
            # Replay-buffer discriminator inputs
            # =========================================================

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

            # =========================================================
            # Discriminator updates
            # =========================================================

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

            running[
                "cycle"
            ] += loss_cycle

            running[
                "feature"
            ] += loss_feature

            running[
                "saliency"
            ] += loss_saliency

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

            running[key] /= (
                denominator
            )

        return running

    # =====================================================
    # Deterministic cycle validation
    # =====================================================

    @torch.no_grad()
    def validate(
        self,
    ):

        self.model.eval()

        # -------------------------------------------------
        # X -> Y -> X
        # -------------------------------------------------

        x_total_loss = 0.0
        x_total_count = 0

        for batch in self.val_x_loader:

            real_x = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
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

            loss = torch.nn.functional.l1_loss(
                reconstructed_x,
                real_x,
                reduction="mean",
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

        # -------------------------------------------------
        # Y -> X -> Y
        # -------------------------------------------------

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

            loss = torch.nn.functional.l1_loss(
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

        val_cycle_mean = (
            val_cycle_x
            + val_cycle_y
        ) / 2.0

        return {
            "val_cycle_x":
                val_cycle_x,

            "val_cycle_y":
                val_cycle_y,

            "val_cycle_mean":
                val_cycle_mean,
        }

    # =====================================================
    # FID / KID evaluation
    # =====================================================

    @torch.no_grad()
    def evaluate_fid_kid(
        self,
    ):
        """
        Forward evaluation:

            validation MAF/DAPI
                -> G_XY
                -> virtual H&E

        compared against held-out real H&E.

        Real H&E metric features are retained.
        Only fake feature state is reset between epochs.
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
            "Evaluating FID/KID..."
        )

        for batch_idx, batch in enumerate(
            self.metric_x_loader
        ):

            real_x = batch[
                "image"
            ].to(
                self.device,
                non_blocking=True,
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

        # -------------------------------------------------
        # Clear fake features, preserve real features.
        # -------------------------------------------------

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

    # =====================================================
    # Preview
    # =====================================================

    @torch.no_grad()
    def save_preview(
        self,
        epoch,
    ):
        """
        Save fixed validation translation previews in the
        PHYSICAL H&E domain.

        During training with invert_y_domain=True:

            G_XY(X) -> inverted H&E

        but the preview should show:

            X -> normal physical H&E

        Likewise, G_YX expects inverted H&E internally, so
        physical real H&E is inverted before reverse
        translation inside this visualization-only wrapper.
        """

        preview_dir = (
            self.preview_dir
            / f"epoch_{epoch:03d}"
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
                """
                X -> model-domain Y -> physical H&E.
                """

                y_model = (
                    self.model.G_XY(
                        x
                    )
                )

                return (
                    trainer
                    ._restore_y_from_model(
                        y_model
                    )
                )

            def G_YX(
                self,
                y_physical,
            ):
                """
                Physical H&E -> model-domain Y -> X.
                """

                y_model = (
                    trainer
                    ._prepare_y_for_model(
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
                self.fixed_val_x,

            y_batch=
                self.fixed_val_y,

            output_dir=
                preview_dir,

            device=
                self.device,

            max_samples=4,
        )

    # =====================================================
    # History
    # =====================================================

    def save_history_row(
        self,
        epoch,
        train_metrics,
        val_metrics,
        distribution_metrics,
        lrs,
        next_lrs,
    ):

        row = {
            "epoch":
                epoch,

            # Backward-compatible alias for generator LR used
            # during this epoch.
            "lr":
                lrs["G"],

            "lr_G":
                lrs["G"],

            "lr_D_X":
                lrs["D_X"],

            "lr_D_Y":
                lrs["D_Y"],

            "next_lr_G":
                next_lrs["G"],

            "next_lr_D_X":
                next_lrs["D_X"],

            "next_lr_D_Y":
                next_lrs["D_Y"],

            "train_G":
                train_metrics[
                    "G"
                ],

            "train_D_X":
                train_metrics[
                    "D_X"
                ],

            "train_D_Y":
                train_metrics[
                    "D_Y"
                ],

            "train_cycle":
                train_metrics[
                    "cycle"
                ],

            "train_feature":
                train_metrics[
                    "feature"
                ],

            "train_saliency":
                train_metrics[
                    "saliency"
                ],

            "val_cycle_x":
                val_metrics[
                    "val_cycle_x"
                ],

            "val_cycle_y":
                val_metrics[
                    "val_cycle_y"
                ],

            "val_cycle_mean":
                val_metrics[
                    "val_cycle_mean"
                ],

            "sub_eval_fid":
                distribution_metrics.get(
                    "fid",
                    np.nan,
                ),

            "sub_eval_kid_mean":
                distribution_metrics.get(
                    "kid_mean",
                    np.nan,
                ),

            "sub_eval_kid_std":
                distribution_metrics.get(
                    "kid_std",
                    np.nan,
                ),
        }

        row_df = pd.DataFrame(
            [
                row
            ]
        )

        if self.history_path.exists():

            old_df = pd.read_csv(
                self.history_path
            )

            # -------------------------------------------------
            # Remove previous row for the same epoch.
            #
            # Useful when an epoch is rerun after a crash.
            # -------------------------------------------------

            if (
                "epoch"
                in old_df.columns
            ):

                old_df = old_df[
                    old_df[
                        "epoch"
                    ]
                    != epoch
                ]

            combined = pd.concat(
                [
                    old_df,
                    row_df,
                ],
                ignore_index=True,
            )

        else:

            combined = row_df

        combined = (
            combined
            .sort_values(
                by="epoch"
            )
            .reset_index(
                drop=True
            )
        )

        combined.to_csv(
            self.history_path,
            index=False,
        )

    # =====================================================
    # Checkpoint utilities
    # =====================================================

    def _checkpoint_payload(
        self,
        epoch,
    ):

        return {
            "epoch":
                epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_G_state_dict":
                self.optimizer_G.state_dict(),

            "optimizer_D_X_state_dict":
                self.optimizer_D_X.state_dict(),

            "optimizer_D_Y_state_dict":
                self.optimizer_D_Y.state_dict(),

            "scheduler_G_state_dict":
                self.scheduler_G.state_dict(),

            "scheduler_D_X_state_dict":
                self.scheduler_D_X.state_dict(),

            "scheduler_D_Y_state_dict":
                self.scheduler_D_Y.state_dict(),

            "scheduler_type":
                self.scheduler_type,

            "epochs_without_fid_improvement":
                self.epochs_without_fid_improvement,

            "config":
                self.config,

            "best_val_cycle_x":
                self.best_val_cycle_x,

            "best_fid":
                self.best_fid,

            "best_fid_epoch":
                self.best_fid_epoch,
        }

    def save_checkpoint(
        self,
        epoch,
        filename,
    ):

        path = (
            self.checkpoint_dir
            / filename
        )

        torch.save(
            self._checkpoint_payload(
                epoch
            ),
            path,
        )

        print(
            "Checkpoint saved:",
            path,
        )

        return path

    # =====================================================
    # Checkpoint cleanup
    # =====================================================

    def _cleanup_old_epoch_checkpoints(
        self,
    ):

        keep_last = int(
            self.checkpoint_cfg.get(
                "keep_last",
                20,
            )
        )

        if keep_last <= 0:
            return

        epoch_files = sorted(
            self.checkpoint_dir.glob(
                "epoch_*.pt"
            )
        )

        if len(
            epoch_files
        ) <= keep_last:

            return

        to_delete = (
            epoch_files[
                :-keep_last
            ]
        )

        for path in to_delete:

            try:

                path.unlink()

                print(
                    "Removed old checkpoint:",
                    path.name,
                )

            except Exception as exc:

                print(
                    "Warning: could not remove "
                    f"{path}: {exc}"
                )

    # =====================================================
    # Load checkpoint
    # =====================================================

    def load_checkpoint(
        self,
        checkpoint_path,
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_absolute():

            checkpoint_path = (
                self.project_root
                / checkpoint_path
            )

        print(
            "Loading checkpoint:",
            checkpoint_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=
                self.device,
            weights_only=False,
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and "model_state_dict"
            in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            isinstance(
                checkpoint,
                dict,
            )
            and "model"
            in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint[
                    "model"
                ]
            )

        else:

            raise RuntimeError(
                "Unsupported checkpoint format. "
                "Expected model_state_dict."
            )

        # -------------------------------------------------
        # Optimizers
        # -------------------------------------------------

        if (
            "optimizer_G_state_dict"
            in checkpoint
        ):

            self.optimizer_G.load_state_dict(
                checkpoint[
                    "optimizer_G_state_dict"
                ]
            )

        if (
            "optimizer_D_X_state_dict"
            in checkpoint
        ):

            self.optimizer_D_X.load_state_dict(
                checkpoint[
                    "optimizer_D_X_state_dict"
                ]
            )

        if (
            "optimizer_D_Y_state_dict"
            in checkpoint
        ):

            self.optimizer_D_Y.load_state_dict(
                checkpoint[
                    "optimizer_D_Y_state_dict"
                ]
            )

        # -------------------------------------------------
        # Schedulers
        #
        # Older teacher checkpoints were produced with LambdaLR.
        # Those scheduler states are intentionally NOT loaded into
        # ReduceLROnPlateau.  The model and Adam optimizer states
        # are still restored, while the new FID-driven scheduler
        # starts fresh from the checkpoint's current optimizer LRs.
        # -------------------------------------------------

        checkpoint_scheduler_type = checkpoint.get(
            "scheduler_type",
            None,
        )

        if checkpoint_scheduler_type is None:
            legacy_state = checkpoint.get(
                "scheduler_G_state_dict",
                {},
            )

            if (
                isinstance(legacy_state, dict)
                and "lr_lambdas" in legacy_state
            ):
                checkpoint_scheduler_type = "legacy_lambda"

        if (
            checkpoint_scheduler_type
            == self.scheduler_type
        ):

            if (
                "scheduler_G_state_dict"
                in checkpoint
            ):
                self.scheduler_G.load_state_dict(
                    checkpoint[
                        "scheduler_G_state_dict"
                    ]
                )

            if (
                "scheduler_D_X_state_dict"
                in checkpoint
            ):
                self.scheduler_D_X.load_state_dict(
                    checkpoint[
                        "scheduler_D_X_state_dict"
                    ]
                )

            if (
                "scheduler_D_Y_state_dict"
                in checkpoint
            ):
                self.scheduler_D_Y.load_state_dict(
                    checkpoint[
                        "scheduler_D_Y_state_dict"
                    ]
                )

            print(
                "Adaptive LR scheduler state restored."
            )

        else:
            print(
                "Scheduler state reset:",
                f"checkpoint={checkpoint_scheduler_type!r},",
                f"current={self.scheduler_type!r}."
            )

            print(
                "Using checkpoint optimizer LRs as the "
                "starting LRs for the new FID-driven scheduler."
            )

        # -------------------------------------------------
        # Best cycle metric
        # -------------------------------------------------

        if (
            "best_val_cycle_x"
            in checkpoint
        ):

            self.best_val_cycle_x = float(
                checkpoint[
                    "best_val_cycle_x"
                ]
            )

        # -------------------------------------------------
        # Best FID
        #
        # Important:
        # old checkpoints may not contain best_fid.
        #
        # In that case KEEP the historical value recovered
        # from fid_kid_results.csv rather than resetting it.
        # -------------------------------------------------

        if (
            "best_fid"
            in checkpoint
            and checkpoint[
                "best_fid"
            ] is not None
            and np.isfinite(
                checkpoint[
                    "best_fid"
                ]
            )
        ):

            checkpoint_best_fid = float(
                checkpoint[
                    "best_fid"
                ]
            )

            self.best_fid = min(
                self.best_fid,
                checkpoint_best_fid,
            )

        if (
            "best_fid_epoch"
            in checkpoint
            and checkpoint[
                "best_fid_epoch"
            ] is not None
        ):

            checkpoint_epoch = int(
                checkpoint[
                    "best_fid_epoch"
                ]
            )

            # Only overwrite if there was no historical one.
            if (
                self.best_fid_epoch
                is None
            ):

                self.best_fid_epoch = (
                    checkpoint_epoch
                )

        if (
            checkpoint_scheduler_type
            == self.scheduler_type
        ):
            self.epochs_without_fid_improvement = int(
                checkpoint.get(
                    "epochs_without_fid_improvement",
                    0,
                )
            )
        else:
            self.epochs_without_fid_improvement = 0

        completed_epoch = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        print(
            "Checkpoint loaded."
        )

        print(
            "Completed epoch:",
            completed_epoch,
        )

        print(
            "Next epoch:",
            completed_epoch + 1,
        )

        print(
            "Best historical FID:",
            (
                f"{self.best_fid:.6f}"
                if np.isfinite(
                    self.best_fid
                )
                else "None"
            ),
        )

        if (
            self.best_fid_epoch
            is not None
        ):

            print(
                "Best historical FID epoch:",
                self.best_fid_epoch,
            )

        self.start_epoch = completed_epoch + 1

        return completed_epoch

    # =====================================================
    # Main training loop
    # =====================================================

    def train(
        self,
        epochs,
        start_epoch=None,
    ):

        num_epochs = int(
            epochs
        )

        if start_epoch is None:
            start_epoch = self.start_epoch

        start_epoch = int(
            start_epoch
        )

        final_epoch = (
            start_epoch
            + num_epochs
            - 1
        )

        print()
        print(
            "=" * 72
        )
        print(
            "STARTING E-CYCLEGAN TRAINING"
        )
        print(
            "=" * 72
        )
        print(
            "Start epoch:",
            start_epoch,
        )
        print(
            "Final epoch:",
            final_epoch,
        )
        print(
            "Train batches:",
            len(
                self.train_loader
            ),
        )
        print(
            "Validation X images:",
            len(
                self.val_x_dataset
            ),
        )
        print(
            "Validation Y images:",
            len(
                self.val_y_dataset
            ),
        )

        save_every = int(
            self.checkpoint_cfg.get(
                "save_every",
                1,
            )
        )

        stopped_early = False

        for epoch in range(
            start_epoch,
            final_epoch + 1,
        ):

            # =================================================
            # Learning rates USED for this epoch
            # =================================================

            epoch_lrs = {
                "G": float(
                    self.optimizer_G
                    .param_groups[0]["lr"]
                ),
                "D_X": float(
                    self.optimizer_D_X
                    .param_groups[0]["lr"]
                ),
                "D_Y": float(
                    self.optimizer_D_Y
                    .param_groups[0]["lr"]
                ),
            }

            # =================================================
            # Train
            # =================================================

            train_metrics = (
                self.train_epoch(
                    epoch
                )
            )

            # =================================================
            # Cycle validation
            # =================================================

            val_metrics = (
                self.validate()
            )

            # =================================================
            # FID / KID
            # =================================================

            if (
                self.fid_enabled
                and epoch
                % self.fid_every
                == 0
            ):
                distribution_metrics = (
                    self.evaluate_fid_kid()
                )
            else:
                distribution_metrics = {
                    "fid": np.nan,
                    "kid_mean": np.nan,
                    "kid_std": np.nan,
                }

            current_fid = float(
                distribution_metrics[
                    "fid"
                ]
            )

            # =================================================
            # Summary for the epoch just completed
            # =================================================

            print()
            print(
                f"Epoch {epoch:03d} summary | "
                f"G={train_metrics['G']:.4f} | "
                f"D_X={train_metrics['D_X']:.4f} | "
                f"D_Y={train_metrics['D_Y']:.4f} | "
                f"train_cycle="
                f"{train_metrics['cycle']:.4f} | "
                f"val_cycle_X="
                f"{val_metrics['val_cycle_x']:.4f} | "
                f"val_cycle_Y="
                f"{val_metrics['val_cycle_y']:.4f} | "
                f"val_mean="
                f"{val_metrics['val_cycle_mean']:.4f}"
            )

            print(
                "LR used | "
                f"G={epoch_lrs['G']:.8f} | "
                f"D_X={epoch_lrs['D_X']:.8f} | "
                f"D_Y={epoch_lrs['D_Y']:.8f}"
            )

            if np.isfinite(
                current_fid
            ):
                print(
                    f"FID={current_fid:.6f} | "
                    f"KID="
                    f"{distribution_metrics['kid_mean']:.8f}"
                    f" ± "
                    f"{distribution_metrics['kid_std']:.8f}"
                )

            # =================================================
            # Preview
            # =================================================

            self.save_preview(
                epoch
            )

            # =================================================
            # Determine FID improvement BEFORE overwriting best
            # =================================================

            is_new_best_fid = False
            is_early_stop_improvement = False

            if np.isfinite(
                current_fid
            ):
                is_new_best_fid = (
                    current_fid
                    < self.best_fid
                )

                is_early_stop_improvement = (
                    current_fid
                    < (
                        self.best_fid
                        - self.early_stopping_min_delta
                    )
                )

            # =================================================
            # Adaptive scheduler
            #
            # Step ONLY when FID was actually computed.
            # =================================================

            if np.isfinite(
                current_fid
            ):
                self.scheduler_G.step(
                    current_fid
                )
                self.scheduler_D_X.step(
                    current_fid
                )
                self.scheduler_D_Y.step(
                    current_fid
                )

            next_lrs = {
                "G": float(
                    self.optimizer_G
                    .param_groups[0]["lr"]
                ),
                "D_X": float(
                    self.optimizer_D_X
                    .param_groups[0]["lr"]
                ),
                "D_Y": float(
                    self.optimizer_D_Y
                    .param_groups[0]["lr"]
                ),
            }

            if any(
                next_lrs[key]
                < epoch_lrs[key]
                for key in (
                    "G",
                    "D_X",
                    "D_Y",
                )
            ):
                print(
                    "Adaptive LR reduction for next epoch:"
                )
                print(
                    f"  G:   {epoch_lrs['G']:.8f} "
                    f"-> {next_lrs['G']:.8f}"
                )
                print(
                    f"  D_X: {epoch_lrs['D_X']:.8f} "
                    f"-> {next_lrs['D_X']:.8f}"
                )
                print(
                    f"  D_Y: {epoch_lrs['D_Y']:.8f} "
                    f"-> {next_lrs['D_Y']:.8f}"
                )

            # =================================================
            # Best forward cycle checkpoint
            # =================================================

            current_val_x = (
                val_metrics[
                    "val_cycle_x"
                ]
            )

            if (
                current_val_x
                < self.best_val_cycle_x
            ):
                self.best_val_cycle_x = (
                    current_val_x
                )

                self.save_checkpoint(
                    epoch=epoch,
                    filename=
                        "best_val_cycle_x.pt",
                )

                print(
                    "New best forward cycle loss:",
                    f"{self.best_val_cycle_x:.6f}",
                )

            # =================================================
            # Best-FID checkpoint
            # =================================================

            if is_new_best_fid:
                old_best = (
                    self.best_fid
                )

                self.best_fid = float(
                    current_fid
                )
                self.best_fid_epoch = (
                    epoch
                )

                if np.isfinite(
                    old_best
                ):
                    print(
                        "New best FID:",
                        f"{old_best:.6f}",
                        "->",
                        f"{self.best_fid:.6f}",
                    )
                else:
                    print(
                        "New best FID:",
                        f"{self.best_fid:.6f}",
                    )

            # =================================================
            # Early-stopping counter
            # =================================================

            if np.isfinite(
                current_fid
            ):
                if is_early_stop_improvement:
                    self.epochs_without_fid_improvement = 0
                else:
                    self.epochs_without_fid_improvement += 1

                print(
                    "FID early-stop counter:",
                    f"{self.epochs_without_fid_improvement}/"
                    f"{self.early_stopping_patience}",
                )

            # Save best-FID AFTER scheduler + early-stop state
            # have been updated, so resume is exact.
            if is_new_best_fid:
                self.save_checkpoint(
                    epoch=epoch,
                    filename=
                        "best_fid.pt",
                )

            # =================================================
            # History
            # =================================================

            self.save_history_row(
                epoch=epoch,
                train_metrics=
                    train_metrics,
                val_metrics=
                    val_metrics,
                distribution_metrics=
                    distribution_metrics,
                lrs=
                    epoch_lrs,
                next_lrs=
                    next_lrs,
            )

            # =================================================
            # Regular epoch checkpoint
            # =================================================

            if (
                epoch % save_every
                == 0
            ):
                self.save_checkpoint(
                    epoch=epoch,
                    filename=
                        f"epoch_{epoch:03d}.pt",
                )
                self._cleanup_old_epoch_checkpoints()

            # =================================================
            # Early stopping
            # =================================================

            if (
                self.early_stopping_patience > 0
                and self.epochs_without_fid_improvement
                >= self.early_stopping_patience
            ):
                print()
                print(
                    "=" * 72
                )
                print(
                    "EARLY STOPPING"
                )
                print(
                    "=" * 72
                )
                print(
                    "No new best FID for",
                    self.epochs_without_fid_improvement,
                    "FID evaluations."
                )
                print(
                    "Best FID:",
                    f"{self.best_fid:.6f}",
                    "at epoch",
                    self.best_fid_epoch,
                )
                stopped_early = True
                break

        print()
        print(
            "Training complete."
            if not stopped_early
            else "Training stopped early."
        )
        print(
            "Best FID:",
            (
                f"{self.best_fid:.6f}"
                if np.isfinite(
                    self.best_fid
                )
                else "None"
            ),
        )

        if (
            self.best_fid_epoch
            is not None
        ):
            print(
                "Best FID epoch:",
                self.best_fid_epoch,
            )


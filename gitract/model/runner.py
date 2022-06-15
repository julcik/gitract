import pytorch_lightning as pl
from typing import Optional, Callable
import pandas as pd
import monai
import torch
from torchmetrics import MetricCollection, ClasswiseWrapper
from gitract.model.metrics import DiceMetric
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image

class LitModule(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float,
            weight_decay: float,
            scheduler: Optional[str],
            T_max: int,
            T_0: int,
            min_lr: int,
            model: str = "unet", #"smpUnet"
    ):
        super().__init__()
        self.classes = ['large_bowel', 'small_bowel', 'stomach']

        self.save_hyperparameters()

        self.model = self._init_model()
        print(self.model)

        self.loss_fn = self._init_loss_fn()

        self.metrics = self._init_metrics()

    def _init_model(self):
        if self.hparams.model == "unet":
            return monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=3,
                channels=(16, 32, 64, 128, 256),
                # channels=(4, 8, 16, 32, 64),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        elif self.hparams.model == "smpFPN":
            return smp.FPN('efficientnet-b2',
                           classes=3,
                           in_channels=3)
        elif self.hparams.model == "smpUnet":
            return smp.Unet('efficientnet-b0',
                           classes=3,
                           in_channels=3)
        elif self.hparams.model == "smpUnetPP":
            return smp.UnetPlusPlus('efficientnet-b2',
                           classes=3,
                           in_channels=3)

    def _init_loss_fn(self):
        # dist_mat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], dtype=np.float32)
        return [#monai.losses.DiceLoss(sigmoid=True),
                monai.losses.DiceFocalLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True, squared_pred=True,
to_onehot_y=False, lambda_dice=0.2),
#                 # monai.losses.GeneralizedWassersteinDiceLoss(dist_mat),
#                 # monai.losses.FocalLoss(gamma=2)
        ]

        # return [smp.losses.TverskyLoss(mode="multilabel",
        #                                classes=None,
        #                                log_loss=True,
        #                                from_logits=True,
        #                                smooth=0.0,
        #                                ignore_index=None,
        #                                eps=1e-07,
        #                                alpha=0.5,
        #                                beta=0.5,
        #                                gamma=1.0)]

    def _init_metrics(self):
        val_metrics = MetricCollection({"val_dice": DiceMetric(classes = self.classes)}, prefix='val/Dice.')
        test_metrics = MetricCollection({"test_dice": DiceMetric(classes = self.classes)})

        return torch.nn.ModuleDict(
            {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.hparams.learning_rate )#,
                                     # weight_decay=self.hparams.weight_decay)

        # if self.hparams.scheduler is None:
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #             optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.min_lr
        #     )
        #
        #     return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        # else:
        return {"optimizer": optimizer}

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def shared_step(self, batch, stage, log=True):
        images, masks = batch["image"], batch["masks"]
        y_pred = self(images)

        loss = sum([func(y_pred, masks) for func in self.loss_fn])

        if stage != "train":
            self.metrics[f"{stage}_metrics"].update(y_pred, masks)

        if log:
            batch_size = images.shape[0]
            self._log(loss, batch_size, None, stage)

        return loss

    def on_validation_epoch_end(self) -> None:
        metric = self.metrics[f"val_metrics"]
        metric_val = metric.compute()
        print(metric_val)
        self.log_dict(metric_val, on_step=False, on_epoch=True)
        metric.reset()

    def _log(self, loss, batch_size, metrics, stage):
        on_step = True if stage == "train" else False

        self.log(f"{stage}/loss", loss, on_step=on_step, prog_bar=True, batch_size=batch_size)

        if metrics is not None:
            self.log_dict(metrics, on_step=on_step, on_epoch=True, batch_size=batch_size)

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module
from typing import Optional
import seaborn as sns
import monai
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torchmetrics import MetricCollection
from torchvision.utils import make_grid

from gitract.model.metrics import DiceMetric


class LitModule(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 1e-3,
            weight_decay: float =0,
            scheduler: Optional[str] = None,
            T_max: int = 1000,
            T_0: int = 0,
            min_lr: int = 0,
            model: str = "unet",  # "smpUnet"
            background: bool = True,
            pretrained: Optional[str] = None,
            slices=5,
    ):
        super().__init__()
        self.classes = ['large_bowel', 'small_bowel', 'stomach']
        self.palette = torch.tensor(sns.color_palette(None, slices), requires_grad=False).T / slices
        self.palette = torch.nn.Parameter(self.palette, requires_grad=False)
        if background:
            self.classes = ['background'] + self.classes

        self.n_classes = len(self.classes)

        self.save_hyperparameters()

        self.model = self._init_model()
        # print(self.model)

        self.loss_fn = self._init_loss_fn()

        self.metrics = self._init_metrics()
        self.post_processing = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )

    def _init_model(self):
        print(f"Init {self.hparams.model}")
        if self.hparams.model == "unet":
            return monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=self.hparams.slices,
                out_channels=self.n_classes,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        elif self.hparams.model == "unetTiny":
            return monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=self.hparams.slices,
                out_channels=self.n_classes,
                channels=(4, 8, 16, 32, 64),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        elif self.hparams.model == "smpFPN":
            return smp.FPN('efficientnet-b2',
                           encoder_weights=self.hparams.pretrained,
                           classes=self.n_classes,
                           decoder_dropout = 0.1,
                           decoder_merge_policy = "cat",
                           in_channels=self.hparams.slices)
        elif self.hparams.model == "smpUnet":
            return smp.Unet('efficientnet-b2',
                            encoder_weights=self.hparams.pretrained,
                            classes=self.n_classes,
                            decoder_attention_type='scse',
                            decoder_channels = [256, 128, 64, 32, 16],
                            in_channels=self.hparams.slices)
        elif self.hparams.model == "smpUnetPP":
            return smp.UnetPlusPlus('efficientnet-b2',
                                    encoder_weights=self.hparams.pretrained,
                                    classes=self.n_classes,
                                    decoder_attention_type='scse',
                                    decoder_channels=[256, 128, 64, 32, 16],
                                    in_channels=self.hparams.slices)
        elif self.hparams.model == "segResNet":
            return monai.networks.nets.SegResNet(
                spatial_dims=2,
                init_filters=8,
                in_channels=self.hparams.slices,
                out_channels=self.n_classes,
                dropout_prob=None,
                act=('RELU', {'inplace': True}),
                norm=('GROUP', {'num_groups': 8}),
                num_groups=8,
                use_conv_final=True,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                upsample_mode="pixelshuffle")
        elif self.hparams.model == "DynUnet":
            # https://github.com/gift-surg/MONAIfbs/blob/main/monaifbs/src/train/monai_dynunet_training.py
            strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
            return monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=self.hparams.slices,
                out_channels=self.n_classes,
                kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                strides=strides,
                upsample_kernel_size=strides[1:],
                filters=None,
                dropout=None,
                norm_name=('INSTANCE', {'affine': True}),
                act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
                deep_supervision=True,
                deep_supr_num=3,
                res_block=False,
                trans_bias=False)
        elif self.hparams.model == "AttentionUnet":
            return monai.networks.nets.AttentionUnet(
                spatial_dims=2,
                in_channels=self.hparams.slices,
                out_channels=self.n_classes,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                kernel_size=3,
                up_kernel_size=3,
                dropout=0.0)
        elif self.hparams.model == "UNETR":
            return monai.networks.nets.UNETR(
                spatial_dims=2,
                in_channels=self.hparams.slices,
                out_channels=self.n_classes,
                img_size=320,
                feature_size=32,
                hidden_size=768,
                mlp_dim=1024,
                num_heads=6, #12
                pos_embed='conv',
                norm_name='instance',
                conv_block=True,
                res_block=True,
                dropout_rate=0.0,)
        elif self.hparams.model == "SwinUNETR":
            return monai.networks.nets.SwinUNETR(
                img_size=(320,320),
                in_channels=self.hparams.slices,
                out_channels=self.n_classes,
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                feature_size=24,
                norm_name='instance',
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                normalize=True,
                use_checkpoint=True,
                spatial_dims=2)

    def _init_loss_fn(self):
        return monai.losses.DiceFocalLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True,
                                          batch=True, squared_pred=True,
                                          to_onehot_y=False)
        # return monai.losses.DiceCELoss(sigmoid=True,
        #                        include_background=True, batch=False, to_onehot_y=False)

        # return smp.losses.TverskyLoss(mode="multilabel",
        #                                classes=None,
        #                                log_loss=True,
        #                                from_logits=True,
        #                                smooth=1.0,
        #                                ignore_index=None,
        #                                eps=1e-07,
        #                                alpha=0.5,
        #                                beta=0.5,
        #                                gamma=1.0)

    def _init_metrics(self):
        val_metrics = MetricCollection(
            {
                "val_dice_classwise": DiceMetric(classes=self.classes),
            }, prefix='val/')
        test_metrics = MetricCollection({"test_dice": DiceMetric(classes=self.classes)})

        return torch.nn.ModuleDict(
            {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.min_lr
            )

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test", batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        pred = self.model(batch.unsqueeze(0))
        return self.post_processing(pred)

    def shared_step(self, batch, stage, batch_idx=None, log=True):
        images, masks = batch["image"], batch["masks"]
        # for i in range(min(images.size(0), 8)):
        #     img = np.array(images[i].detach().cpu().moveaxis(0,-1)*0.224 + 0.456)*255
        #     mask = np.array(masks[i].detach().cpu().moveaxis(0,-1))*255
        #     Image.fromarray(np.hstack((img, mask)).astype(np.uint8)).save(f"{str(batch_idx)}_{str(i)}.png")

        y_pred = self.model(images)
        if stage=="train" and hasattr(self.model, "deep_supervision") and self.model.deep_supervision:
            masks = masks.unsqueeze(1).repeat([1,y_pred.size(1),1,1,1])
        loss = self.loss_fn(y_pred, masks)

        if stage != "train":
            self.metrics[f"{stage}_metrics"].update(y_pred, masks)

        if log:
            batch_size = images.shape[0]
            self._log(loss, batch_size, stage)


        if batch_idx == 1:
            y_pred = torch.sigmoid(y_pred)

            self.loggers[1].experiment.add_image(f"{stage}/prediction",
                                                 torch.Tensor.cpu(make_grid(
                                                     [
                                                         torch.Tensor.cpu(
                                                             make_grid([torch.tensordot(self.palette, images[b], dims=([1], [0])),
                                                                        masks[b][-3:]] +
                                                                       [y_pred[b, clz].unsqueeze(0).repeat([3, 1, 1])
                                                                        for clz in range(self.n_classes)],
                                                                       nrow=2 + self.n_classes)
                                                         )
                                                         for b in range(min(8, images.size(0)))], nrow=1)
                                                 ), self.current_epoch, dataformats="CHW")

        return loss

    def on_validation_epoch_end(self) -> None:
        metric = self.metrics[f"val_metrics"]
        metric_val = metric.compute()
        print(metric_val)
        self.log_dict(metric_val, on_step=False, on_epoch=True)
        metric.reset()

    def _log(self, loss, batch_size, stage):
        on_step = True if stage == "train" else False

        self.log(f"{stage}/loss", loss, on_step=on_step, prog_bar=True, batch_size=batch_size)

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module

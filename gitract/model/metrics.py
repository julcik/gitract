import monai
from monai.utils import MetricReduction
from torchmetrics import Metric
import torch


class DiceMetric(Metric):
    def __init__(self, classes, prefix="Dice"):
        super().__init__()

        self.classes = classes
        self.prefix = prefix

        self.post_processing = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )
        self.metric = monai.metrics.DiceMetric(include_background=True,
                                               reduction=MetricReduction.MEAN_BATCH,
                                               get_not_nans=False,
                                 )

    def update(self, y_pred, y_true):
        y_pred = self.post_processing(y_pred)
        self.metric(y_pred, y_true)


    def compute(self):
        self.metric.reduction = MetricReduction.MEAN_BATCH
        metric = self.metric.aggregate()
        classwise = {f"{self.prefix}.{self.classes[clz]}":metric[clz] for clz in range(len(self.classes))}

        self.metric.reduction = MetricReduction.MEAN
        mean = self.metric.aggregate()
        classwise.update({f"m{self.prefix}": mean})
        return classwise

    def reset(self) -> None:
        self.metric.reset()


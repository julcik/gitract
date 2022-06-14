import monai
from torchmetrics import Metric
import torch

class DiceMetric(Metric):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes

        self.post_processing = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )
        self.add_state("dice", default=[])

    def update(self, y_pred, y_true):
        y_pred = self.post_processing(y_pred)
        self.dice.extend(monai.metrics.compute_meandice(y_pred, y_true))

    def compute(self):
        metric = torch.mean(torch.nan_to_num(torch.stack(self.dice)), dim=0)
        return {self.classes[clz]:metric[clz] for clz in range(len(self.classes))}
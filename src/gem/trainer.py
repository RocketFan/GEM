import segmentation_models_pytorch as smp
import torch.nn as nn
import torchmetrics

from torchgeo.trainers import SemanticSegmentationTask


class FocalDiceLoss(nn.Module):
    def __init__(
        self, mode: str = "multiclass", ignore_index: int = 0, normalized: bool = False
    ):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(
            mode=mode, ignore_index=ignore_index, normalized=normalized
        )
        self.dice_loss = smp.losses.DiceLoss(mode=mode, ignore_index=ignore_index)

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets) + self.dice_loss(preds, targets)


class DFC2022SemanticSegmentationTask(SemanticSegmentationTask):
    def configure_losses(self) -> None:
        if self.hparams["loss"] == "ce":
            self.criterion = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
                ignore_index=-1000 if self.hparams["ignore_index"] is None else 0
            )
        elif self.hparams["loss"] == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif self.hparams["loss"] == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=self.hparams["ignore_index"], normalized=True
            )
        elif self.hparams["loss"] == "focaldice":
            self.criterion = FocalDiceLoss(
                mode="multiclass",
                ignore_index=self.hparams["ignore_index"],
                normalized=True,
            )
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def configure_metrics(self) -> None:
        task = "multiclass"
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "OverallAccuracy": torchmetrics.Accuracy(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    multidim_average="global",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "OverallPrecision": torchmetrics.Precision(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    multidim_average="global",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "OverallRecall": torchmetrics.Recall(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    multidim_average="global",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "AverageAccuracy": torchmetrics.Accuracy(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    multidim_average="global",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "AveragePrecision": torchmetrics.Precision(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    multidim_average="global",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "AverageRecall": torchmetrics.Recall(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    multidim_average="global",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "IoU": torchmetrics.JaccardIndex(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    ignore_index=self.hparams["ignore_index"],
                ),
                "F1Score": torchmetrics.FBetaScore(
                    task=task,
                    num_classes=self.hparams["num_classes"],
                    beta=1.0,
                    average="micro",
                    multidim_average="global",
                    ignore_index=self.hparams["ignore_index"],
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

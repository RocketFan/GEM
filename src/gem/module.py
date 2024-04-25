import lightning as L
import torch

from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from gem.model import SegNet


class EarthMapping(L.LightningModule):
    def __init__(self, features=32, n_classes=16, lr=0.001, class_weights: Tensor | None = None) -> None:
        super().__init__()

        self.lr = lr
        self.class_weights = class_weights
        self.n_classes = n_classes
        self.model = SegNet(features=features, out_channels=n_classes)

    def forward(self, batch):
        inputs = batch['image'][:, :3]
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs = batch['image'][:, :3]
        masks = self.transform_mask(batch['mask'])

        outputs = self.model(inputs)

        loss_fn = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(outputs, masks)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['image'][:, :3]
        masks = self.transform_mask(batch['mask'])

        outputs = self.model(inputs)

        loss_fn = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(outputs, masks)

        self.log('val_loss', loss)

    def transform_mask(self, mask: Tensor):
        mask = one_hot(mask.long(), num_classes=self.n_classes)
        mask = mask.squeeze(1).permute(0, 3, 1, 2).float()

        return mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

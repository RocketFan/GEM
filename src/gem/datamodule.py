import kornia.augmentation as K
import lightning as pl
import torch
import torchvision.transforms as T

from einops import rearrange
from torch.utils.data import DataLoader
from torchgeo.datamodules.utils import dataset_split
from gem.dataset import DFC2022Dataset

DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["input", "mask"],
)


class DFC2022DataModule(pl.LightningDataModule):
    # Stats computed in labeled train set
    dem_min, dem_max = -79.18, 3020.26
    dem_nodata = -99999.0

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        patch_size: int | None = None,
        predict_on: str = "val",
        augmentations=DEFAULT_AUGS,
        img_size=2048,
        n_tiles=1,
        **kwargs,
    ):
        super().__init__()
        assert predict_on in DFC2022Dataset.metadata
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.patch_size = patch_size
        self.predict_on = predict_on
        self.augmentations = augmentations
        self.img_size = img_size
        self.patch_size = patch_size

        if self.patch_size is not None:
            self.random_crop = K.AugmentationSequential(
                K.RandomCrop((self.patch_size, self.patch_size), p=1.0, keepdim=False),
                data_keys=["input", "mask"],
            )

            transforms = T.Compose([self.preprocess, self.crop])
        else:
            transforms = T.Compose([self.preprocess])

        self.dataset = DFC2022Dataset(self.root_dir, "train", transforms=transforms, img_size=img_size, n_tiles=n_tiles)

    def preprocess(self, sample):
        sample["image"][:3] /= 255.0
        sample["image"][-1] = (sample["image"][-1] - self.dem_min) / (self.dem_max - self.dem_min)
        sample["image"][-1] = torch.clip(sample["image"][-1], min=0.0, max=1.0)

        if "mask" in sample:
            sample["mask"][sample["mask"] == 15] = 0
            sample["mask"] = sample["mask"].to(torch.long)
        
        return sample

    def crop(self, sample):
        sample["mask"] = sample["mask"].to(torch.float)
        sample["image"], sample["mask"] = self.random_crop(
            sample["image"], sample["mask"]
        )
        sample["mask"] = sample["mask"].to(torch.long)
        sample["image"] = sample['image'].squeeze(0)
        sample["mask"] = sample['mask'].squeeze()
        return sample

    def setup(self, stage=None):
        test_transforms = T.Compose([self.preprocess])

        self.train_dataset, self.val_dataset, _ = dataset_split(
            self.dataset, val_pct=self.val_split_pct, test_pct=0.0
        )
        self.predict_dataset = DFC2022Dataset(
            self.root_dir, self.predict_on, transforms=test_transforms, img_size=self.img_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            if self.augmentations is not None:
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                batch["mask"] = batch["mask"].to(torch.long)
                batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")

        return batch

import rasterio
import numpy as np

from collections.abc import Callable
from torch import Tensor
from torchvision.transforms import v2
from torchgeo.datasets import DFC2022


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def get_tile(tensor, x, y, tile_size):
    return tensor[:, x:x + tile_size, y:y + tile_size]


class DFC2022Dataset(DFC2022):
    def __init__(self, root: str = "data",
                 split: str = "train",
                 transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
                 checksum: bool = False,
                 n_tiles=1,
                 img_size=2000):

        resize_transform = v2.Compose([
            v2.Resize((img_size, img_size)),
        ])

        if transforms is None:
            transforms = resize_transform
        else:
            transforms = v2.Compose([resize_transform, transforms])

        super().__init__(root, split, transforms, checksum)
        self.n_tiles = n_tiles
        self.tile_size = img_size // n_tiles
        self.img_size = img_size

        self.__filter_unlabeled()

    def get_tile_size(self):
        return self.tile_size

    def calc_class_percentages(self):
        class_counts = np.zeros(len(self.classes))

        for file in self.files:
            mask = rasterio.open(file['target']).read()

            unique_labels, counts = np.unique(mask, return_counts=True)

            for label, count in zip(unique_labels, counts):
                class_counts[label.item()] += count.item()

        total_samples = len(self.files) * self.img_size ** 2

        return class_counts / total_samples

    def __getitem__(self, index):
        file_index = index // (self.n_tiles**2)

        x = (index % self.n_tiles) * self.tile_size
        y = ((index // self.n_tiles) % self.n_tiles) * self.tile_size

        data = super().__getitem__(file_index)

        image = get_tile(data['image'][:3], x, y, self.tile_size)
        mask = get_tile(data['mask'].unsqueeze(0), x, y, self.tile_size)

        return dict(image=image, mask=mask)

    def __len__(self):
        return len(self.files) * (self.n_tiles ** 2)

    def __filter_unlabeled(self):
        labeled_files = [file for file in self.files if 'target' in file]

        for file in labeled_files:
            label = rasterio.open(file['target']).read(out_shape=(1, 1, 30))
            unique_labels = np.unique(label)
            if unique_labels.all() == 0:
                self.files.remove(file)

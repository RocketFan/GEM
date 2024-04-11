import torch
import os

from skimage import io
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MiniFranceDataset(Dataset):
    def __init__(self, minifrance_geo, data_type="", transform=None):
        self.minifrance_geo = self.load_data_type(minifrance_geo, data_type)
        self.transform_image = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((2000, 2000)),
                transforms.PILToTensor(),
            ]
        )

        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        minifrance_row = self.minifrance_geo.iloc[index]

        img_path = os.path.join(minifrance_row["region_dir_path"], "BDORTHO",
                                str(minifrance_row.name) + ".tif")
        image = io.imread(img_path, plugin="pil")
        image = self.transform_image(image)

        dem_path = os.path.join(minifrance_row["region_dir_path"], "RGEALTI",
                                str(minifrance_row.name) + "_RGEALTI" + ".tif")
        dem = io.imread(dem_path, plugin="pil")

        landcover_map_path = os.path.join(minifrance_row["region_dir_path"], "UrbanAtlas", 
                                          str(minifrance_row.name) + "_UA2012" + ".tif")
        landcover_map = io.imread(landcover_map_path, plugin="pil")
        landcover_map = self.transform_image(landcover_map)

        geodata = minifrance_row.drop('geometry').to_dict()

        if self.transform:
            image = self.transform(image)
            dem = self.transform(dem)
            landcover_map = self.transform(landcover_map)

        return {"image": image, "dem": dem, "landcover_map": landcover_map, "geodata": geodata}

    def __len__(self):
        return len(self.minifrance_geo)

    def load_data_type(self, minifrance_geo, data_type: str):
        if not data_type:
            return minifrance_geo
        else:
            return minifrance_geo[minifrance_geo["data_type"] == data_type]

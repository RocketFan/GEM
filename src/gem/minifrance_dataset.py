import torch
import os
import pandas as pd
import torchvision.transforms as transforms
import time
import rasterio

from skimage import io
from torch.utils.data import Dataset


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def get_tile(tensor, x, y, tile_size):
    return tensor[:, x:x + tile_size, y:y + tile_size]


class MiniFranceDataset(Dataset):
    def __init__(self, dir_path: str, img_size=2048, dataset_type: str = 'all', n_tiles=1):
        self.n_tiles = n_tiles
        self.tile_size = img_size // n_tiles
        self.dataset_type = dataset_type
        self.img_size = img_size

        self.file_df = self.__create_file_dataframe(dir_path)

    def get_tile_size(self):
        return self.tile_size

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        file_index = index // (self.n_tiles**2)
        file_df_row = self.file_df.iloc[file_index]

        x = (index % self.n_tiles) * self.tile_size
        y = ((index // self.n_tiles) % self.n_tiles) * self.tile_size

        image = io.imread(file_df_row['image_path'], plugin="pil")
        image = rasterio.open(file_df_row['image_path']).read(out_shape=(3, self.img_size, self.img_size))
        image = torch.from_numpy(image)
        image = get_tile(image, x, y, self.tile_size)

        dem = rasterio.open(file_df_row['dem_path']).read(out_shape=(1, self.img_size, self.img_size))
        dem = torch.from_numpy(dem)
        dem = get_tile(dem, x, y, self.tile_size)

        if file_df_row['lc_path']:
            landcover_map = rasterio.open(file_df_row['lc_path']).read(out_shape=(1, self.img_size, self.img_size)) 
            landcover_map = torch.from_numpy(landcover_map)
        else:
            landcover_map = torch.zeros_like(image)
        landcover_map = get_tile(landcover_map, x, y, self.tile_size)

        return {"image": image, "dem": dem, "landcover_map": landcover_map,
                "region": file_df_row['region']}

    def __len__(self):
        return len(self.file_df) * (self.n_tiles ** 2)

    def __create_file_dataframe(self, dir_path: str):
        file_list = []

        if self.dataset_type == 'all':
            for dataset_type in os.listdir(dir_path):
                files = self.__get_all_dataset_type_files(dir_path, dataset_type)
                file_list += files
        else:
            file_list = self.__get_all_dataset_type_files(dir_path, self.dataset_type)

        return pd.DataFrame(file_list)

    def __get_all_dataset_type_files(self, dir_path, dataset_type) -> list[dict]:
        type_path = os.path.join(dir_path, dataset_type)
        file_list = []

        if not os.path.isdir(type_path):
            return file_list

        for region in os.listdir(type_path):
            region_path = os.path.join(type_path, region)
            if not os.path.isdir(region_path):
                continue

            image_dir = os.path.join(region_path, 'BDORTHO')
            for image_file in os.listdir(image_dir):
                record = self.__create_record(dataset_type, region, region_path, image_dir, image_file)

                if record:
                    file_list.append(record)

        return file_list

    def __create_record(self, dataset_type, region, region_path, image_dir, image_file):
        if image_file.endswith('.tif'):
            base_filename = os.path.splitext(image_file)[0]
            dem_path = os.path.join(
                region_path, 'RGEALTI', f'{base_filename}_RGEALTI.tif')
            lc_path = os.path.join(
                region_path, 'UrbanAtlas', f'{base_filename}_UA2012.tif')

            if os.path.exists(dem_path):
                return {
                    'image_path': os.path.join(image_dir, image_file),
                    'dem_path': dem_path,
                    'lc_path': lc_path if os.path.exists(lc_path) else None,
                    'region': region,
                    'dataset_type': dataset_type
                }
            else:
                return None

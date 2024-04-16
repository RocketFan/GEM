import torch
import os
import pandas as pd
import torchvision.transforms as transforms

from skimage import io
from torch.utils.data import Dataset


class MiniFranceDataset(Dataset):
    def __init__(self, dir_path: str, dataset_type: str = 'all'):
        self.dataset_type = dataset_type
        self.transform_image = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((2048, 2048)),
                transforms.PILToTensor(),
            ]
        )

        self.file_df = self.__create_file_dataframe(dir_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        file_df_row = self.file_df.iloc[index]

        image = io.imread(file_df_row['image_path'], plugin="pil")
        image = self.transform_image(image)

        dem = io.imread(file_df_row['dem_path'], plugin="pil")
        dem = self.transform_image(dem)

        landcover_map = io.imread(file_df_row['lc_path'], plugin="pil")
        landcover_map = self.transform_image(landcover_map)

        return {"image": image, "dem": dem, "landcover_map": landcover_map,
                "region": file_df_row['region']}

    def __len__(self):
        return len(self.file_df)

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

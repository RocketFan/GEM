import os
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.warp

from shapely.geometry import Point
from shapely import wkt


class MiniFranceTIFData:
    def __init__(self, dir_path) -> None:
        self.dir_path = dir_path
        self.crs = "EPSG:3857"
        self.geodata_filename = "geodata.csv"
        self.geodata_path = os.path.join(self.dir_path, self.geodata_filename)

    def read_geodata(self):
        coords_df = self.load_or_generate_df()
        coords_geo = gpd.GeoDataFrame(coords_df, index=coords_df.index, geometry=coords_df.geometry)
        coords_geo.set_crs(self.crs, inplace=True, allow_override=True)
        return coords_geo

    def load_or_generate_df(self):
        if os.path.exists(self.geodata_path):
            return self.load_df()

        coords_df = self.generate_df()
        coords_df["geometry"] = coords_df.coors.apply(lambda x: Point(x))
        coords_df.to_csv(self.geodata_path)

        return coords_df

    def load_df(self):
        coords_df = pd.read_csv(self.geodata_path, index_col=0)
        coords_df["geometry"] = coords_df.geometry.apply(wkt.loads)
        return coords_df

    def generate_df(self):
        coords_df = pd.DataFrame(columns=["coors", "geometry", "region", "data_type", "region_dir_path"])

        for root, _, files in os.walk(self.dir_path):
            if "BDORTHO" in root:
                for filename in files:
                    if filename.endswith(".tif"):
                        lon, lat = self.extract_coords_from_file(os.path.join(root, filename))
                        region, data_type, region_dir_path = self.extract_info_from_dir_path(root)
                        index = self.extract_index(filename)

                        coords_df.loc[index] = {
                            "coors": [lon, lat],
                            "data_type": data_type,
                            "region": region,
                            "region_dir_path": region_dir_path,
                        }

        return coords_df

    def extract_coords_from_file(self, path):
        with rasterio.open(path) as f:
            height, width = f.shape
            center_i, center_j = height // 2, width // 2
            center_x, center_y = f.transform * (center_j + 0.5, center_i + 0.5)
            center_geom = rasterio.warp.transform_geom(f.crs, self.crs, {"type": "Point", "coordinates": (center_x, center_y)}, precision=6)
            return center_geom["coordinates"]

    def extract_info_from_dir_path(self, path: str):
        path_elements = path.split("/")
        region = path_elements[-2]
        data_type = path_elements[-3]
        region_dir_path = "/".join(path_elements[:-1])
        return region, data_type, region_dir_path

    def extract_index(self, filename: str):
        filename_elements = filename.split(".")
        fielename_without_extension = filename_elements[0]
        return fielename_without_extension


if __name__ == "__main__":
    directory = "/GEM/data/MiniFrance"

    minifrance_data = MiniFranceTIFData(directory)
    df = minifrance_data.read_geodata()
    print(df.head())

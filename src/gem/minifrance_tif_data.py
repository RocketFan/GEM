import os
import rasterio
import rasterio.features
import rasterio.warp
import geopandas as gpd
import pandas as pd

from shapely import Point

class MiniFranceTIFData:
    def __init__(self, dir_path) -> None:
        self.dir_path = dir_path
        self.crs = 'EPSG:3857'

    def read_geodata(self):
        coords_df = pd.DataFrame(columns=["coors"])
        i = 0

        for root, dirs, files in os.walk(self.dir_path):
            if "BDORTHO" not in root:
                continue

            for filename in files:
                if ".tif" not in filename:
                    continue

                lon, lat = self.extract_coords_from_file(os.path.join(root, filename))
                coords_df.loc[len(coords_df)] = {'coors': [lon, lat]}
                i += 1

        coords_df["geometry"] = coords_df.coors.apply(lambda x: Point(x))
        coords_geo = gpd.GeoDataFrame(coords_df)
        coords_geo.set_crs(self.crs, inplace=True, allow_override=True)

        return coords_geo
    
    def extract_coords_from_file(self, path):
        with rasterio.open(path) as f:
            height, width = f.shape
            center_i, center_j = height // 2, width // 2
            center_x, center_y = f.transform * (center_j + 0.5, center_i + 0.5)
            center_geom = rasterio.warp.transform_geom(f.crs, self.crs, {'type': 'Point', 'coordinates': (center_x, center_y)}, precision=6)

            return center_geom["coordinates"]

if __name__ == "__main__":
    directory = "/GEM/data/labeled_train"

    minifrance_data = MiniFranceTIFData(directory)
    df = minifrance_data.read_geodata()
    print(df.head())

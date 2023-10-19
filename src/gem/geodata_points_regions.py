import geopandas as gpd
import matplotlib.pyplot as plt

class GeodataPointsRegions:
    def __init__(self, coords_geo, france_geo):
        self.coords_geo = coords_geo
        self.france_geo = france_geo

    def get_regions_coverage(self):
        points_geo = self.coords_geo.copy()
        points_geo.geometry = self.coords_geo.buffer(10000)

        return points_geo.dissolve(by="region")

    def get_interseting_regions(self, regions_coverage_geo):
        intersecting_regions = []

        for index, polygon in self.france_geo.iterrows():
            region_intersections = regions_coverage_geo.intersects(polygon.geometry)

            if region_intersections.any():
                intersecting_region = self.__get_intersecting_region(regions_coverage_geo, polygon, region_intersections)
                intersecting_regions.append(intersecting_region)

        intersecting_regions = self.__filter_regions_with_coverage(intersecting_regions, 0.4)

        return intersecting_regions

    def __get_intersecting_region(self, regions_coverage_geo, polygon, region_intersections):
        coverage_areas = []
        for _, row in regions_coverage_geo[region_intersections].iterrows():
            intersection = row.geometry.intersection(polygon["geometry"])
            area = intersection.area
            coverage_areas.append({"area": area, "geodata": row, "intersection": intersection})

        coverage_region = max(coverage_areas, key=lambda x: x["area"])

        return {
            "geometry": polygon["geometry"],
            "point_coverage": coverage_region["area"],
            "region": coverage_region["geodata"].index,
            "data_type": coverage_region["geodata"].data_type,
        }

    def __filter_regions_with_coverage(self, regions, coverage_percentage):
        intersecting_gdf = gpd.GeoDataFrame(regions)
        intersecting_gdf.set_crs(self.france_geo.crs, inplace=True)
        intersecting_gdf["polygon_area"] = intersecting_gdf.geometry.area
        intersecting_gdf["point_coverage_percentage"] = (intersecting_gdf["point_coverage"] / intersecting_gdf["polygon_area"])

        return intersecting_gdf[intersecting_gdf["point_coverage_percentage"] >= coverage_percentage]
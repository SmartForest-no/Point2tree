import geopandas as gpd
import pdal
import json
import subprocess
import glob
import os
from pathlib import Path

class DensityFiltering:
    def __init__(self, path_data, min_density=1, count_threshold=15000, buffer_size=0.01, verbose=False):
        self.path_data = Path(path_data)
        # remove the extension
        self.path_data_out = os.path.join(os.path.dirname(path_data), os.path.basename(path_data).split('.')[0])
        self.min_density = min_density
        self.count_threshold = count_threshold
        self.buffer_size = buffer_size
        self.verbose = verbose
        # get a file with suffix .shp in self.mls_boundary_path
        self.mls_boundary_path = None

    def compute_density(self):
        # check if the file exists
        if not os.path.exists(self.path_data):
            raise Exception(f"File {self.path_data} does not exist.")
        cmd_density = f"pdal density {self.path_data} {self.path_data_out} --threshold {self.min_density}"
        subprocess.call(cmd_density, shell=True)
        self.mls_boundary_path = glob.glob(os.path.join(self.path_data_out, '*.shp'))[0]

    def filter_and_dissolve_boundary_shapefile(self):
        # check if the file exists
        if not os.path.exists(self.mls_boundary_path):
            raise Exception(f"File {self.mls_boundary_path} does not exist.")

        gdf = gpd.read_file(self.mls_boundary_path)
        filtered = gdf[gdf['COUNT'] > self.count_threshold]
        # check if the filtered is empty
        if filtered.empty:
            print("The filtered shapefile is empty. No clipping will be done.")
            return None
        else:
            dissolved = filtered.buffer(self.buffer_size).unary_union
            dissolved_gdf = gpd.GeoDataFrame(geometry=[dissolved])
            dissolved_gdf['CLASS'] = 30
            dissolved_gdf.to_file(self.mls_boundary_path)
            return dissolved_gdf

    def clip_point_cloud(self):
        mls_path_clipped = os.path.join(self.path_data_out, "_clipped.las")

        pipeline_clipping_MLS = [
            str(self.path_data),
            {
                "type": "filters.overlay",
                "dimension": "Classification",
                "datasource": self.mls_boundary_path,
                "column": "CLASS"
            },
            {
                "type": "filters.range",
                "limits": "Classification[30:30]"
            },
            {
                "filename": mls_path_clipped
            }
        ]

        pipeline = pdal.Pipeline(json.dumps(pipeline_clipping_MLS))
        pipeline.execute()

    def process(self):
        self.compute_density()
        # check density if self.mls_boundary_path is not empty 
        if self.mls_boundary_path is None:
            raise Exception("self.mls_boundary_path is None. Check if self.compute_density() was called.")
        result = self.filter_and_dissolve_boundary_shapefile()
        if result is not None:
            self.clip_point_cloud()
            if self.verbose:
                print(f"File {self.path_data} has been processed.")
                print(f"Clipped file is in {os.path.join(self.path_data_out, '_clipped.las')}")
            return os.path.join(self.path_data_out, '_clipped.las')
        else:
            if self.verbose:
                print(f"File {self.path_data} has not been processed.")
                print(f"No clipping was done because the filtered shapefile is empty.This means that the number of points in the file is less than {self.count_threshold}.")
            return None
     
if __name__ == '__main__':
    # read the parameters from the command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--min_density', type=int, required=True)
    parser.add_argument('--count_threshold', type=int, required=True)
    parser.add_argument('--buffer_size', type=float, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    processor = DensityFiltering(args.path_data, args.min_density, args.count_threshold, args.buffer_size, args.verbose)
    processor.process()

    # a simple way to test the code
    # python density_filtering.py 
    # --path_data /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/new_density_data/first.las 
    # --min_density 1 
    # --count_threshold 15000 --buffer_size 0.01
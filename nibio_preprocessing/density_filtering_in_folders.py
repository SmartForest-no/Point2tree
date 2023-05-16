import glob
import os
import argparse
import shutil
from nibio_preprocessing.density_filtering import DensityFiltering

class DensityFilteringInFolders:
    def __init__(self, folder, min_density, count_threshold, buffer_size, verbose=False):
        self.input_folder = folder
        self.min_density = min_density
        self.count_threshold = count_threshold
        self.buffer_size = buffer_size
        self.verbose = verbose

    def filter_density_in_folder(self):
        # get all the files in the folder
        files = glob.glob(os.path.join(self.input_folder, '*.las'))
        for file in files:
            print(f"Processing file {file}")
            density_filtering = DensityFiltering(file, self.min_density, self.count_threshold, self.buffer_size, self.verbose)
            result = density_filtering.process()
          
            # copy files from os.path.join(self.path_data_out, "_clipped.las") to the input folder and rename them as the original file
            if result is not None:
                shutil.move(
                    os.path.join(density_filtering.path_data_out, "_clipped.las"), 
                    os.path.join(file))
        
            
if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description='Filter the density of the point cloud in a folder.')
    parser.add_argument('--input_folder', type=str, required=True, help='The folder containing the point clouds to be filtered.')
    parser.add_argument('--min_density', type=int, required=True, help='The minimum density of the point cloud.')
    parser.add_argument('--count_threshold', type=int, required=True, help='The minimum number of points in the filtered point cloud.')
    parser.add_argument('--buffer_size', type=float, required=True, help='The buffer size for the boundary shapefile.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # run the density filtering
    density_filtering_in_folders = DensityFilteringInFolders(
        args.input_folder, 
        args.min_density, 
        args.count_threshold, 
        args.buffer_size, 
        args.verbose
        )
    
    density_filtering_in_folders.filter_density_in_folder()

    # a simple way to test the code
    # python nibio_preprocessing/density_filtering_in_folders.py --input_folder /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/check_maciek/ --min_density 1 --count_threshold 15000 --buffer_size 0.01 --verbose 
        




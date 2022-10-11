import argparse
from functools import reduce
import os
from tabnanny import verbose
import laspy # requires laspy==2.1.2 (check if doesn't work)
import numpy as np
from tqdm import tqdm


class ShrinkPointCloud():
    def __init__(self, input_folder, output_folder, treshold=0.9, verbose=False):
        self.directory_with_point_clouds = input_folder
        self.output_folder = output_folder
        self.threshold = 1.0 - float(treshold)
        self.verbose = verbose
    
    def reduce_point_cloud(self, point_cloud, reduce_z=False):
        # get the points from the point cloud
        min_x = point_cloud.header.min[0]
        max_x = point_cloud.header.max[0]
        min_y = point_cloud.header.min[1]
        max_y = point_cloud.header.max[1]
        min_z = point_cloud.header.min[2]
        max_z = point_cloud.header.max[2]

        original_points_count = point_cloud.header.point_count

        point_cloud = point_cloud[np.logical_and(point_cloud.x > min_x * self.threshold, point_cloud.x < max_x * self.threshold)]
        point_cloud = point_cloud[np.logical_and(point_cloud.y > min_y * self.threshold, point_cloud.y < max_y * self.threshold)]
        if reduce_z:
            point_cloud = point_cloud[np.logical_and(point_cloud.z > min_z * self.threshold, point_cloud.z < max_z * self.threshold)]

        reduced_points_count = point_cloud.header.point_count
        if self.verbose:
            print("Reduced point cloud from {} to {} points.".format(original_points_count, reduced_points_count))
            print("Reduced point cloud by {}%.".format((1 - reduced_points_count / original_points_count) * 100))
        
        return point_cloud

    def reduce_point_clouds(self):
        # get paths to all point clouds in the directory and subdirectories
        point_cloud_paths = []

        # create output folder if it doesn't exist, if it does, delete all files in it
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        else:
            files = os.listdir(self.output_folder)
            for f in files:
                os.remove(os.path.join(self.output_folder, f))


        for root, dirs, files in os.walk(self.directory_with_point_clouds):
            for file in files:
                if file.endswith(".las"):
                    point_cloud_paths.append(os.path.join(root, file))
        if self.verbose:
            print("Found {} point clouds.".format(len(point_cloud_paths)))
        # iterate over all point clouds and save outputs to the output folder
        for point_cloud_path in tqdm(point_cloud_paths):
            # load the point cloud
            inFile = laspy.read(point_cloud_path)
            # reduce the point cloud to the desired density
            inFile = self.reduce_point_cloud(inFile)
            # save the point cloud to the output folder
            # inFile.write(os.path.join(self.output_folder, os.path.basename(point_cloud_path)))
            inFile.write(os.path.join(self.output_folder, os.path.basename(point_cloud_path).replace(".las", "_reduced.las")))

    
        if self.verbose:
            print("Done.")
            print("Saved to {}.".format(self.output_folder))
           
    def main(self):
        self.reduce_point_clouds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="The folder where the point clouds are stored.")
    parser.add_argument("--output_folder", help="The folder where the reduced point clouds will be stored.")
    parser.add_argument("--threshold", help="The threshold for the reduction. Default is 0.9.", default=0.9)
    parser.add_argument("--verbose", help="Prints more information.", action="store_true")
    args = parser.parse_args()
    shrink_point_cloud = ShrinkPointCloud(args.input_folder, args.output_folder, args.threshold, args.verbose)
    shrink_point_cloud.main()




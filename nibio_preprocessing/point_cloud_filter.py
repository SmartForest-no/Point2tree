import argparse
from functools import reduce
import os
from tabnanny import verbose
import laspy # requires laspy==2.1.2 (check if doesn't work)
import numpy as np
from tqdm import tqdm


class PointCloudFilter():
    def __init__(self, directory_with_point_clouds, density=50, in_place=False, verbose=False):
        self.directory_with_point_clouds = directory_with_point_clouds
        self.verbose = verbose
        self.in_place = in_place
        self.density = density
    
    def compute_density(self, point_cloud):
        # find the dimensions of the point cloud
        min_x = point_cloud.header.min[0]
        max_x = point_cloud.header.max[0]
        min_y = point_cloud.header.min[1]
        max_y = point_cloud.header.max[1]
        min_z = point_cloud.header.min[2]
        max_z = point_cloud.header.max[2]

        # compute the dimensions of the point cloud
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z

        # compute the number of points in the point cloud
        num_points = point_cloud.header.point_count

        # compute the density of the point cloud
        density = num_points / (x_range * y_range * z_range)

        # return the density
        return density

    def reduce_point_cloud(self, point_cloud, threshold, reduce_z=False):
        # get the points from the point cloud
        min_x = point_cloud.header.min[0]
        max_x = point_cloud.header.max[0]
        min_y = point_cloud.header.min[1]
        max_y = point_cloud.header.max[1]
        min_z = point_cloud.header.min[2]
        max_z = point_cloud.header.max[2]

        original_points_count = point_cloud.header.point_count

        point_cloud = point_cloud[np.logical_and(point_cloud.x > min_x * threshold, point_cloud.x < max_x * threshold)]
        point_cloud = point_cloud[np.logical_and(point_cloud.y > min_y * threshold, point_cloud.y < max_y * threshold)]
        if reduce_z:
            point_cloud = point_cloud[np.logical_and(point_cloud.z > min_z * threshold, point_cloud.z < max_z * threshold)]

        reduced_points_count = point_cloud.header.point_count
        if self.verbose:
            print("Reduced point cloud from {} to {} points.".format(original_points_count, reduced_points_count))
            print("Reduced point cloud by {}%.".format((1 - reduced_points_count / original_points_count) * 100))
        
        return point_cloud

    def reduce_point_cloud_to_density(self, inFile, reduce_z=False):
        original_point_cloud_count = inFile
        for i in range(20, 1, -1):
            outFile = self.reduce_point_cloud(original_point_cloud_count, i / 20, reduce_z)
            print("Current point clound density: ", self.compute_density(outFile))
            if self.compute_density(outFile) > self.density:
                print("Final point clound density: ", self.compute_density(outFile))
                return outFile

    def main(self):
        # get paths to all point clouds in the directory and subdirectories
        point_cloud_paths = []
        for root, dirs, files in os.walk(self.directory_with_point_clouds):
            for file in files:
                if file.endswith(".las"):
                    point_cloud_paths.append(os.path.join(root, file))
        if self.verbose:
            print("Found {} point clouds.".format(len(point_cloud_paths)))
        # iterate over all point clouds and save outputs to the same directory
        for point_cloud_path in tqdm(point_cloud_paths):
            # load the point cloud
            inFile = laspy.read(point_cloud_path)
            # reduce the point cloud to the desired density
            inFile = self.reduce_point_cloud_to_density(inFile)
            # save the point cloud to the same directory with a new name
            inFile.write(point_cloud_path.replace(".las", "_filtered.las"))

            # do write in place if specified
            if self.in_place:
                os.remove(point_cloud_path)
                os.rename(point_cloud_path.replace(".las", "_filtered.las"), point_cloud_path)

        if self.verbose:
            print("Done.")
            print("Saved to {}.".format(os.path.join(self.directory_with_point_clouds, "filtered_point_clouds")))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='', help='path to the folder containing the point clouds')
    parser.add_argument('--density', type=int, default=50, help='desired density')
    parser.add_argument('--in_place', action='store_true', default=False, help='write in place')
    parser.add_argument('--reduce-z', action='store_true', default=False,  help='reduce z as well as x and y')
    parser.add_argument('--verbose', action='store_true', default=False,  help='print more information')
    args = parser.parse_args()
    point_cloud_filter = PointCloudFilter(args.dir, args.density, args.verbose)
    point_cloud_filter.main()

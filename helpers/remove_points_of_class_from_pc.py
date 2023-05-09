import argparse
import glob
import os
import laspy
import numpy as np

class RemovePointsOfClassFromPC:
    def __init__(self, folder_name, class_name, class_value, verbose=False):
        self.folder_name = folder_name
        self.class_name = class_name
        self.class_value = class_value
        self.verbose = verbose

    def get_paths_of_files(self, folder_name):
        # use glob to get all the paths of the files in the folder
        paths = glob.glob(os.path.join(folder_name, "*.las"), recursive=False)

        # check if the folder is empty
        if len(paths) == 0:
            raise Exception("The folder is empty")

        if self.verbose:
            print("The number of files in the folder {} is {}".format(folder_name, len(paths)))

        return paths

    def read_one_las_file_and_remove_points(self, file_path, class_name, class_value):
        # read the las file
        las_file = laspy.read(file_path)

        # check if at least one point has the class value in the las file
        if np.sum(las_file[class_name] == class_value) == 0:
            if self.verbose:
                print("No points with the class value {} in the file {}".format(class_value, file_path))
                print("No points removed from the file {}".format(file_path))


        # create a new las file with the same header as the original las file
        new_file = laspy.create(point_format=las_file.header.point_format, file_version=las_file.header.version)

        # write the points to the las file except the points of the class
        new_file.points = las_file.points[las_file[class_name] != class_value]

        # write the las file
        new_file.write(file_path)

    def remove_points_of_class_from_pc(self, folder_name, class_name, class_value):
        # get the paths of all the files in the folder
        paths = self.get_paths_of_files(folder_name)

        # read all the files and remove the points of the class
        for path in paths:
            if self.verbose:
                print("Removing points from the file {}".format(path))
            self.read_one_las_file_and_remove_points(path, class_name, class_value)

        if self.verbose:
            print("Done")

    def __call__(self):
        self.remove_points_of_class_from_pc(self.folder_name, self.class_name, self.class_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove points of a class from a point cloud")
    parser.add_argument("-f", "--folder_name", type=str, help="The folder name where the point clouds are stored")
    parser.add_argument("-n", "--class_name", type=str, help="The class name of the points to be removed")
    parser.add_argument("-c", "--class_value", type=int, help="The class value of the points to be removed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the progress")

    args = parser.parse_args()

    # get the arguments
    folder_name = args.folder_name
    class_name = args.class_name
    class_value = args.class_value
    verbose = args.verbose

    # remove the points of the class from the point cloud
    remove_points_of_class_from_pc = RemovePointsOfClassFromPC(folder_name, class_name,class_value, verbose)
    remove_points_of_class_from_pc()


        
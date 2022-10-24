import argparse
from email import header
import glob
import os
import laspy
import numpy as np

#TODO: remove ground if exists
#TODO: add meriging of files into one file (maybe using pdal)


class GetInstancesSideBySide():
    def __init__(self, input_folder, output_folder, instance_label='instance_nr', verbose=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.instance_label = instance_label
        self.verbose = verbose

    def process_single_file(self, file_path):
        if self.verbose:
            print("Processing file: {}".format(file_path))
        las_file = laspy.read(file_path)
        # get file header version and point format
        header = las_file.header
        point_format = las_file.point_format
        # get the points
        points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
        # get the instance labels
        instance_labels = las_file[self.instance_label]
        # get the unique instance labels
        unique_instance_labels = np.unique(instance_labels)
        # create a dictionary to store the points for each instance
        instance_points = {}
        # iterate over the unique instance labels
        for instance_label in unique_instance_labels:
            # get the points for the current instance
            instance_points[instance_label] = points[instance_labels == instance_label]
        return instance_points, header, point_format

    def process_folder(self):
        # get all files in the folder
        files = glob.glob(self.input_folder + '/*.las')
        # create the output folder if it does not exist
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        # process each file
        for file in files:
            instance_points, header, point_format = self.process_single_file(file)
            # get the new box coordinates
            new_mean_coordinates = self.get_new_coordinates(file)
        
            # save files to separate files
            for instance_label, points in instance_points.items():
                # creat a new las file
                new_header = laspy.LasHeader(point_format=point_format.id, version=header.version)
                las = laspy.LasData(new_header)
            
                # get box coordinates
                min_x = np.min(points[:, 0])
                min_y = np.min(points[:, 1])
                min_z = np.min(points[:, 2])
                # zero the coordinates
                points[:, 0] = points[:, 0] - min_x
                points[:, 1] = points[:, 1] - min_y
                points[:, 2] = points[:, 2] - min_z

                # add the new coordinates
                points[:, 0] = points[:, 0] + new_mean_coordinates[instance_label]['x_aligned']
                points[:, 1] = points[:, 1] + new_mean_coordinates[instance_label]['y_aligned']

                # add the points to the las file
                las.x = points[:, 0]
                las.y = points[:, 1]
                las.z = points[:, 2]
                # write the las file to the output folder
                las.write(os.path.join(self.output_folder, str(instance_label) + '.las'))
                if self.verbose:
                    print("Saved instance {} to file".format(instance_label))


    def get_new_coordinates(self, file_path):
        instance_points, _, _ = self.process_single_file(file_path)

        instance_coordinates = {}

        for instance_label, points in instance_points.items():
            # get box coordinates
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            min_z = np.min(points[:, 2])
            max_z = np.max(points[:, 2])
            # zero the coordinates
            points[:, 0] = points[:, 0] - min_x
            points[:, 1] = points[:, 1] - min_y
            points[:, 2] = points[:, 2] - min_z
            # get the new box coordinates
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            min_z = np.min(points[:, 2])
            max_z = np.max(points[:, 2])
            # get mean coordinates x, and y
            mean_x = np.mean(points[:, 0])
            mean_y = np.mean(points[:, 1])
            # put all to the dictionary
            param={}
            param['min_x'] = min_x
            param['max_x'] = max_x
            param['min_y'] = min_y
            param['max_y'] = max_y
            param['min_z'] = min_z
            param['max_z'] = max_z
            param['mean_x'] = (max_x - min_x) /2    
            param['mean_y'] = (max_y - min_y) /2
            instance_coordinates[instance_label] = param

        # compute new coordinates
        new_instance_coordinates = {}
        # get a global mean value of all y coordinates
        y_aligned = np.mean([instance_coordinates[instance_label]['mean_y'] for instance_label in instance_coordinates])
        # all zeros

        y_aligned = 0
        # add it all the y coordinates in new_instance_coordinates
        for i, instance_label in enumerate(instance_coordinates):
            new_instance_coordinates[instance_label] = {}
            new_instance_coordinates[instance_label]['y_aligned'] = y_aligned
            # new_instance_coordinates[instance_label]['x_aligned'] = 0.5 * y_aligned + 0.5 * instance_coordinates[instance_label]['mean_x']
            if i == 0:
                new_instance_coordinates[instance_label]['x_aligned'] = instance_coordinates[instance_label]['mean_x']
            else:
                new_instance_coordinates[instance_label]['x_aligned'] = \
                new_instance_coordinates[instance_label-1]['x_aligned'] \
                + 1.0 * (instance_coordinates[instance_label-1]['max_x'] - (instance_coordinates[instance_label-1]['min_x'])) \
                + 1.0 * (instance_coordinates[instance_label]['max_x'] - (instance_coordinates[instance_label]['min_x'])) 

        return new_instance_coordinates

if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser(
        description='Get instances side by side')
    parser.add_argument('--input_folder', dest='input_folder',
                        type=str, help='Input folder', default='./data/')  
    parser.add_argument('--output_folder', dest='output_folder',
                        type=str, help='Output folder', default='./output/')
    parser.add_argument('--instance_label', dest='instance_label',
                        type=str, help='Instance label', default='instance_nr')
    parser.add_argument('--verbose', action='store_true', help="Print information about the process")
    args = parser.parse_args()

    # create instance of GetInstancesSideBySide
    get_instances = GetInstancesSideBySide(
        args.input_folder, 
        args.output_folder,
        args.instance_label, 
        args.verbose)
    get_instances.process_folder()


 
    
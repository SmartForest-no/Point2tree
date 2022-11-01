import argparse
from email import header
import glob
import os
from unicodedata import name
import laspy
import numpy as np
import pandas as pd
import pickle


class GetInstancesSideBySide():
    def __init__(self, input_folder, output_folder, instance_label='instance_nr', verbose=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.instance_label = instance_label
        self.verbose = verbose
        self.las_file_header_version = str(1.2) # can be changed in the future 
        self.las_file_point_format_id = 3 # can be changed in the future
        self.stats_file = os.path.join(self.output_folder, 'stats_' + str(self.instance_label) + '.csv')
      
    def process_single_file(self, file_path):
        las_file = laspy.read(file_path)
        instance_labels = las_file[self.instance_label]
        points = np.vstack((las_file.x, las_file.y, las_file.z, instance_labels)).transpose()
        # get the unique instance labels
        unique_instance_labels = np.unique(instance_labels)
        # create a dictionary to store the points for each instance
        instance_points = {}
        # iterate over the unique instance labels
        for instance_label in unique_instance_labels:
            # get the points for the current instance
            instance_points[instance_label] = points[instance_labels == instance_label]
        return instance_points

    def process_folder(self):
        # get all files in the folder
        files = glob.glob(self.input_folder + '/*.las', recursive=False)

        # create the output folder if it does not exist
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        # process each file

        # create a new csv file using pandas
        df = pd.DataFrame(columns=[
            'file_name', 
            'instance_label', 
            'min_x', 
            'max_x', 
            'min_y', 
            'max_y', 
            'min_z', 
            'max_z', 
            'number_of_points'
            ])
        
        # create a dictionary to store the points for each instance
        low_ground_points_dict = {}

        for file in files:

            instance_points = self.process_single_file(file)
            # get the new box coordinates
            new_mean_coordinates = self.get_new_coordinates(file)
        
            # save files to separate files
            for instance_label, points in instance_points.items():
                # creat a new las file
                new_header = laspy.LasHeader(point_format=self.las_file_point_format_id, version=self.las_file_header_version)
                new_header.add_extra_dim(laspy.ExtraBytesParams(name=str(self.instance_label), type=np.int32))
                las = laspy.LasData(new_header)
            
                # get box coordinates
                min_x = np.min(points[:, 0])
                min_y = np.min(points[:, 1])
                min_z = np.min(points[:, 2])

                max_x = np.max(points[:, 0])
                max_y = np.max(points[:, 1])
                max_z = np.max(points[:, 2])

                # define the new coordinates of a size of the old one 
                points_low_ground = np.zeros_like(points)
                # get points to z dimension of 0.5 meter
                points_low_ground[:, 2] = points[:, 2] - min_z
                # get only the points which are lower than 0.1 meter
                points_low_ground = points_low_ground[points_low_ground[:, 2] < 0.3]
                # save the points to the 

                # add parameters to the dataframe
                df = df.append(
                    {
                    'file_name': os.path.basename(file), 
                    'instance_label': instance_label, 
                    'min_x': min_x, 
                    'max_x': max_x, 
                    'min_y': min_y, 
                    'max_y': max_y, 
                    'min_z': min_z, 
                    'max_z': max_z, 
                    'number_of_points': len(points)
                    }, ignore_index=True)

                # add the points to the dictionary               
                low_ground_points_dict[str(instance_label)] = points_low_ground

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
                las[str(self.instance_label)] = points[:, 3]

                # write the las file to the output folder
                las.write(os.path.join(self.output_folder, str(instance_label) + '.las'))
                # write csv file
                df.to_csv(self.stats_file, index=False)
            
            # save dictionary to pickle file
            with open(os.path.join(self.output_folder, 'low_ground_points_dict.pickle'), 'wb') as handle:
                pickle.dump(low_ground_points_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            if self.verbose:
                # print the number of instances which were saved and done
                print("Saved {} instances".format(len(instance_points)))
                print("Done with file: {}".format(file))

    def remove_all_files_in_output_folder_exept_merged(self):
        # get all files in the folder
        files = glob.glob(self.output_folder + '/*.las')
        for file in files:
            if file != os.path.join(self.output_folder, 'merged_' + str(self.instance_label) + '.las'):
                os.remove(file)

        if self.verbose:
            print("Done with removing all files exept merged")

    def remove_all_files_in_output_folder(self):
        # get all files in the folder
        files = glob.glob(self.output_folder + '/*.las')
        for file in files:
            os.remove(file)

        if self.verbose:
            print("Done with removing all files")

    def merge_all_files(self):

        # remove merged file if it exists
        if os.path.isfile(os.path.join(self.output_folder, 'merged_' + str(self.instance_label) + '.las')):
            os.remove(os.path.join(self.output_folder, 'merged_' + str(self.instance_label) + '.las'))
            print("Removed file: {}".format(os.path.join(self.output_folder, 'merged_' + str(self.instance_label) + '.las')))

        # get all files in the folder
        files = glob.glob(self.output_folder + '/*.las')
        # create a new las file
        new_header = laspy.LasHeader(point_format=self.las_file_point_format_id,  version=self.las_file_header_version)
        new_header.add_extra_dim(laspy.ExtraBytesParams(name=str(self.instance_label), type=np.int32))
        las = laspy.LasData(new_header)
        tmp_dict = {}
        small_subset = ['X', 'Y', 'Z', str(self.instance_label)]
        for item in small_subset:
            tmp_dict[item] = []

        for file in files:
            las_file = laspy.read(file)
            for item in list(small_subset):
                tmp_dict[item] = np.append(tmp_dict[item], las_file[item])
  
        for key in tmp_dict.keys():
            las[key] = tmp_dict[key]
        # write the las file to the output folder and add suffix merged and self.instance_label
        las.write(os.path.join(self.output_folder, 'merged_' + str(self.instance_label) + '.las'))
        if self.verbose:
            print("Done with merging all files")

    def get_new_coordinates(self, file_path):
        instance_points = self.process_single_file(file_path)

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
        instance_nr_list = list(instance_coordinates.keys())
        # add it all the y coordinates in new_instance_coordinates
        for i, instance_label in enumerate(instance_nr_list):
            new_instance_coordinates[instance_label] = {}
            new_instance_coordinates[instance_label]['y_aligned'] = y_aligned
            # new_instance_coordinates[instance_label]['x_aligned'] = 0.5 * y_aligned + 0.5 * instance_coordinates[instance_label]['mean_x']
            if i == 0:
                new_instance_coordinates[instance_label]['x_aligned'] = instance_coordinates[instance_label]['mean_x']
            else:
                new_instance_coordinates[instance_label]['x_aligned'] = \
                new_instance_coordinates[instance_nr_list[i - 1]]['x_aligned'] \
                + 1.0 * (instance_coordinates[instance_nr_list[i - 1]]['max_x'] - (instance_coordinates[instance_nr_list[i - 1]]['min_x'])) \
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
    parser.add_argument('--merge', action='store_true', help="Print information about the process")
    parser.add_argument('--verbose', action='store_true', help="Print information about the process")
    args = parser.parse_args()

    # create instance of GetInstancesSideBySide
    get_instances = GetInstancesSideBySide(
        args.input_folder, 
        args.output_folder,
        args.instance_label, 
        args.verbose)

    # get_instances.remove_all_files_in_output_folder()
    get_instances.process_folder()
    if args.merge:
        get_instances.merge_all_files()
        get_instances.remove_all_files_in_output_folder_exept_merged()


 
    
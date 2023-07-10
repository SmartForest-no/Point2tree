# Author: Maciej Wielgosz 
# THis the oracle wrapper for the Oracle Cloud Infrastructure (OCI) environment and Docker container

import argparse
import os
import shutil

import yaml

from run import main

DEBUG_MODE = False

def run_oracle_wrapper(path_to_config_file):

    # read the config file from config folder
    with open(path_to_config_file) as f:
        config_flow_params = yaml.load(f, Loader=yaml.FullLoader)

    # read system environment variables

    if DEBUG_MODE:
        input_location = "/home/bucket/local_folder_oracle_hard/las_files/"
        output_location = "/home/bucket/local_folder_oracle_hard/results/"

        # input_location = "/home/bucket/local_folder_oracle/las_files/"
        # output_location = "/home/bucket/local_folder_oracle/results/"

        # remove content of the output folder if it exists
        if os.path.exists(output_location):
            shutil.rmtree(output_location, ignore_errors=True)
    else:
        # get the input and output locations from the environment variables
        input_location = os.environ['OBJ_INPUT_LOCATION']
        output_location = os.environ['OBJ_OUTPUT_LOCATION']

        # remap the input and output locations
        input_location = input_location.replace("@axqlz2potslu", "").replace("oci://", "/mnt/")
        output_location = output_location.replace("@axqlz2potslu", "").replace("oci://", "/mnt/")

    # create the output folder if it does not exist
    os.makedirs(output_location, exist_ok=True)


    # copy files from input_location to the input folder
    shutil.copytree(input_location, config_flow_params['general']['input_folder'])

    # run the main function
    main(path_to_config_file)

    # instance segmentation is set to true
    if config_flow_params['general']['run_instance_segmentation']:
        path_to_the_output_folder = os.path.join(config_flow_params['general']['output_folder'], 'instance_segmented_point_clouds_with_ground')
    else:
        path_to_the_output_folder = config_flow_params['general']['output_folder']

    # zip the files in path_to_the_output_folder
    zip_file_name  = 'results'
    shutil.make_archive(zip_file_name, 'zip', path_to_the_output_folder) # this will be done in the current folder
    shutil.copy('results.zip', path_to_the_output_folder)

    # copy the zip file and other files to the output location
    for filename in os.listdir(path_to_the_output_folder):
        src_file = os.path.join(path_to_the_output_folder, filename)
        dst_file = os.path.join(output_location, filename)
        shutil.copy(src_file, dst_file)

if __name__ == '__main__':
    # use argparse to get the path to the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config_file", type=str, default="./config/config.yaml")
    args = parser.parse_args()

    # run the main function
    print('Running the main function in run_oracle_wrapper.py')
    run_oracle_wrapper(args.path_to_config_file)


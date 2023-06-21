# This is the the file to be run on the oracle cloud

import oci
import argparse
import os
import io
import sys
import json
import shutil
import yaml
from urllib.parse import urlparse
from pathlib import Path
from oci.config import validate_config
from oci.object_storage import ObjectStorageClient


def run_oracle_wrapper(path_to_config_file):
    # read the config file with the credentials with json format
    with open('login_oracle_config.json') as f:
        config = json.load(f)

    # validate the config file
    validate_config(config)

    # create the client
    client = ObjectStorageClient(config)

    # read system environment variables
    input_location = os.environ['OBJ_INPUT_LOCATION']
    output_location = os.environ['OBJ_OUTPUT_LOCATION']

    # input_location = "oci://maciej-seg-test-in@axqlz2potslu/las_files"
    # output_location = "oci://maciej-seg-test-out@axqlz2potslu"

    # input_location = "oci://forestsens_temp@axqlz2potslu/acc_6/batch_274/original_las_files"
    # output_location = "oci://maciej-seg-test-out@axqlz2potslu"


    # doing for the input
    if input_location is not None:
        print('Taking the input from the location ' + input_location)
        parsed_url = urlparse(input_location)
        input_folder_in_bucket = parsed_url.path[1:]
        input_bucket_name = parsed_url.netloc.split('@')[0]
        input_namespace = parsed_url.netloc.split('@')[1]

        print("input_folder_in_bucket: ", input_folder_in_bucket)
        print("input_bucket_name: ", input_bucket_name)
        print("input_namespace: ", input_namespace)

    else:
        print('Taking the input from the default location')
        # get the input_namespace
        input_namespace = client.get_input_namespace().data
        # get the bucket name
        input_bucket_name = 'bucket_lidar_data'
        # folder name inside the bucket
        input_folder_in_bucket = 'geoslam'

    # doing for the output
    if output_location is not None:
        print('Saving the output to the location ' + output_location)
        parsed_url = urlparse(output_location)
        output_folder_in_bucket = parsed_url.path[1:]
        output_bucket_name = parsed_url.netloc.split('@')[0]
        output_namespace = parsed_url.netloc.split('@')[1]

    else:
        print('Saving the output to the default location')
        # get the output_namespace
        output_namespace = client.get_input_namespace().data
        # get the bucket name
        output_bucket_name = 'bucket_lidar_data'
        # folder name inside the bucket
        output_folder_in_bucket = 'output'

    # read the config file from config folder
    with open(path_to_config_file) as f:
        config_flow_params = yaml.load(f, Loader=yaml.FullLoader)

    # copy all files from the bucket to the input folder
    # get the list of objects in the bucket
    list_objects_response = client.list_objects(
        namespace_name=input_namespace,
        bucket_name=input_bucket_name,
        prefix=input_folder_in_bucket,
        limit=1000)

    # objects = client.list_objects(input_namespace, input_bucket_name).data.objects
    objects = list_objects_response.data.objects
    # create a list of objects which contain also files not just folders
    objects = [item for item in objects if item.name[-1] != '/']

    # create the input folder if it does not exist
    if not os.path.exists(config_flow_params['general']['input_folder']):
        os.mkdir(config_flow_params['general']['input_folder'])

    # download the files from the bucket to the input folder
    for item in objects:
        object_name = item.name
        print("object name: ", object_name)

        # Get the object's data
        file = client.get_object(input_namespace, input_bucket_name, object_name)

        # Use only the base name of the object for the local file
        local_file_name = os.path.basename(object_name)

        # Open the local file
        with open(local_file_name, 'wb') as f:
            for chunk in file.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        # remove the file if it already exists
        if os.path.exists(os.path.join(config_flow_params['general']['input_folder'], object_name)):
            os.remove(os.path.join(config_flow_params['general']['input_folder'], object_name))

        # move the file to the input folder and overwrite if it already exists
        shutil.move(local_file_name, config_flow_params['general']['input_folder'])

    from run import main

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

    # get list of files in the output folder
    list_of_files = os.listdir(path_to_the_output_folder)

    # save files to the output bucket 'bucket_lidar_data' in the subfolder 'output'
    for file in list_of_files:
        # get the full path of the file
        path_to_file = os.path.join(path_to_the_output_folder, file)

        # upload the file to the bucket
        client.put_object(
            output_namespace, 
            output_bucket_name, 
            os.path.join(output_folder_in_bucket, file), 
            io.open(path_to_file, 'rb')
            )

if __name__ == '__main__':
    # use argparse to get the path to the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config_file", type=str, default="./config/config.yaml")
    args = parser.parse_args()

    # run the main function
    print('Running the main function in run_oracle_wrapper.py')
    run_oracle_wrapper(args.path_to_config_file)


# This is the the file to be run on the oracle cloud

import oci
import argparse
import os
import io
import sys
import json
import shutil
import yaml
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

    # get the namespace
    namespace = client.get_namespace().data

    # get the bucket name
    bucket_name = 'bucket_lidar_data'

    # get the object name
    object_name = './geoslam/plot72_tile_-25_-25.las'

    # get the object
    file = client.get_object(namespace, bucket_name, object_name)

    # write the object to a file
    with open('plot_from_bucket.las', 'wb') as f:
        for chunk in file.data.raw.stream(1024 * 1024, decode_content=False):
            f.write(chunk)


   # read the config file from config folder
    with open(path_to_config_file) as f:
        config_flow_params = yaml.load(f, Loader=yaml.FullLoader)

    # create the input folder if it does not exist
    if not os.path.exists(config_flow_params['general']['input_folder']):
        os.mkdir(config_flow_params['general']['input_folder'])

    # move the file to the input folder using shutil if it does not exist
    if not os.path.exists(config_flow_params['general']['input_folder'] + '/plot_from_bucket.las'):
        shutil.move('plot_from_bucket.las', config_flow_params['general']['input_folder'])


    from run import main

    # run the main function
    main(path_to_config_file)

if __name__ == '__main__':
    # use argparse to get the path to the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config_file", type=str, default="./config/config.yaml")
    args = parser.parse_args()

    # run the main function
    run_oracle_wrapper(args.path_to_config_file)


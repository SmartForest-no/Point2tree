# This script is used to run the application in a production environment
import argparse
import os
import yaml
import logging

# local imports
from helpers.run_command_bash import RunCommandBash

# define logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(path_to_config_file):
    # load the config file
    with open(path_to_config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # check if the output folder exists and if not create it
    if not os.path.exists(config["general"]["output_folder"]):
        os.mkdir(config["general"]["output_folder"])

    ### sematic segmentation section ###
    if config["general"]["run_sematic_segmentation"]:
        logger.info("Running semantic segmentation")
        sem_seg_command = config["semantic_segmentation_params"]["sematic_segmentation_script"]

        # print the semantic segmentation parameters
        for key, value in config["semantic_segmentation_params"].items():
            logger.info(key + ": " + str(value))

        # read all the parameters from the config file for the semantic segmentation
        sem_seg_args = []

        sem_seg_args.extend([
            "-d", str(config["general"]["input_folder"]),
            "-c", str(config["semantic_segmentation_params"]["checkpoint_model_path"]),
            "-b", str(config["semantic_segmentation_params"]["batch_size"]),
            "-t", str(config["semantic_segmentation_params"]["tile_size"]),
            "-m", str(config["semantic_segmentation_params"]["min_density"]),
            "-z", str(config["semantic_segmentation_params"]["remove_small_tiles"])
            ])

        # run the command with the arguments
        logging.info("Running semantic segmentation with the arguments")
        RunCommandBash(sem_seg_command, sem_seg_args)()

    ### instance segmentation section ###
    if config["general"]["run_instance_segmentation"]:
        logger.info("Running instance segmentation")
        ins_seg_command = config["instance_segmentation_params"]["instance_segmentation_script"]

        # print the instance segmentation parameters
        for key, value in config["instance_segmentation_params"].items():
            logger.info(key + ": " + str(value))

        # read all the parameters from the config file for the instance segmentation
        ins_seg_args = []

        ins_seg_args.extend([
        "-d", str(config["general"]["input_folder"]),
        "-n", str(config["instance_segmentation_params"]["n_tiles"]),
        "-s", str(config["instance_segmentation_params"]["slice_thickness"]),
        "-h", str(config["instance_segmentation_params"]["find_stems_height"]),
        "-t", str(config["instance_segmentation_params"]["find_stems_thickness"]),
        "-g", str(config["instance_segmentation_params"]["graph_maximum_cumulative_gap"]),
        "-l", str(config["instance_segmentation_params"]["add_leaves_voxel_length"]),
        "-m", str(config["instance_segmentation_params"]["find_stems_min_points"]),
        "-o", str(config["instance_segmentation_params"]["graph_edge_length"]),
        "-p", str(config["instance_segmentation_params"]["add_leaves_edge_length"])
        ])

        # run the command with the arguments
        logging.info("Running instance segmentation with the arguments")
        RunCommandBash(ins_seg_command, ins_seg_args)()

    # do cleaning up folders
    if config["general"]["clean_output_folder"]:
        logger.info("Cleaning up the output folder")
        os.system("rm -rf {}".format(config["general"]["output_folder"]))
        os.mkdir(config["general"]["output_folder"])
    else:
        # check if the output folder is empty
        if len(os.listdir(config["general"]["output_folder"])) != 0:
            logger.error("The output folder is not empty. Please clean it up or set the 'clean_output_folder' parameter to True")
            exit(1)

    ### if only semantic segmentation is run transfer data to the output folder
    if config["general"]["run_sematic_segmentation"] and not config["general"]["run_instance_segmentation"]:
        logger.info("Transfering data to the output folder for semantic segmentation")
        source_dir = os.path.join(config["general"]["input_folder"], "segmented_point_clouds") 

        # take paths of all the files in the source_dir which end with .segmented.ply
        files = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if file.endswith(".segmented.ply")]
    
        # convert files in the source_dir to .las using pdal
        for input_file in files:
            # get directory of the file
            dir_name = os.path.dirname(input_file)
            # get file name
            file_name = os.path.basename(input_file).split(".")[0]
            # create a new file name with '.segmented.las' at the end
            output_file_name = os.path.join(dir_name, file_name + ".segmented.las")

            # create the command
            os.system("pdal translate {} {} --writers.las.dataformat_id=3 --writers.las.extra_dims=all".format(input_file, output_file_name))
        
        # copy the converted files to the output folder
        las_segmented_files = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if file.endswith(".segmented.las")]
        for file in las_segmented_files:
            os.system("cp {} {}".format(file, config["general"]["output_folder"]))

    ### if both semantic and instance segmentation are run transfer data to the output folder
    if config["general"]["run_sematic_segmentation"] and config["general"]["run_instance_segmentation"]:
        source_folder = os.path.join(config["general"]["input_folder"], "results")
        # copy all the files and folders from the source folder to the output folder
        os.system("cp -r {} {}".format(source_folder + '/*', config["general"]["output_folder"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run the application in a production environment.')
    parser.add_argument("--path_to_config_file", type=str, default="./config/config.yaml")
    args = parser.parse_args()

    # run the main function
    main(args.path_to_config_file)
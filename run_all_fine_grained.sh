#!/bin/bash

############################ parameters #################################################
# General parameters
CLEAR_INPUT_FOLDER=1  # 1: clear input folder, 0: not clear input folder
CONDA_ENV="pdal-env-1" # conda environment for running the pipeline

# Tiling parameters
N_TILES=3
SLICE_THICKNESS=0.5
FIND_STEMS_HEIGHT=1.5
FIND_STEMS_THICKNESS=0.5
GRAPH_MAXIMUM_CUMULATIVE_GAP=3
ADD_LEAVES_VOXEL_LENGTH=0.5
FIND_STEMS_MIN_POINTS=50
############################# end of parameters declaration ############################


# Do the environment setup
# check if PYTHONPATH is set to the current directory
if [ -z "$PYTHONPATH" ]; then
    echo "PYTHONPATH is not set. Setting it to the current directory"
    export PYTHONPATH=$PWD
else
    echo "PYTHONPATH is set to '$PYTHONPATH'"
fi

# conda activate pdal-env-1

# check if activated conda environment is the same as the one specified in the parameters
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    echo "The activated conda environment is not the same as the one specified in the parameters."
    echo "Please activate the correct conda environment and run the script again."
    exit 1
fi

data_folder=$1

# if no input folder is provided, case a message and exit
if [ -z "$data_folder" ]
then
    echo "No input folder provided, please provide the input folder as a command line argument"
    exit 1
fi

# clear input folder if CLEAR_INPUT_FOLDER is set to 1
if [ $CLEAR_INPUT_FOLDER -eq 1 ]
then
    # delete all the files and folders except the ply, las and laz files in the input folder
    echo "Clearing input folder"
    find $data_folder/ -type f ! -name '*.ply' ! -name '*.las' ! -name '*.laz' -delete # delete all the files except the ply and las files
    find $data_folder/* -type d -exec rm -rf {} + # delete all the folders in the input folder
    echo "Removed all the files and folders except the ply and las files in the input folder"
fi

# check if there are las and laz files in the input folder
count_las=`ls -1 $data_folder/*.las 2>/dev/null | wc -l`
count_laz=`ls -1 $data_folder/*.laz 2>/dev/null | wc -l`

count=$(($count_las + $count_laz))

if [ $count != 0 ]; then
    echo "$count las files found in the input folder good to go with!"
else
    echo "No las or laz files found in the input folder."
    echo "All files in the input folder should have *.las or *.laz extension."
    exit 1
fi 

# do the conversion from laz to las if there are laz files in place (this is need for metrics calculation)
python nibio_preprocessing/convert_files_in_folder.py --input_folder $data_folder --output_folder $data_folder --out_file_type las --in_place --verbose

# do the conversion to ply
python nibio_preprocessing/convert_files_in_folder.py --input_folder $data_folder --output_folder $data_folder --out_file_type ply --verbose

# clear input folder if CLEAR_INPUT_FOLDER is set to 1
if [ $CLEAR_INPUT_FOLDER -eq 1 ]
then
    # delete all the files and folders except the ply and las files in the input folder
    echo "Clearing input folder"
    find $data_folder/ -type f ! -name '*.ply' ! -name '*.las' -delete # delete all the files except the ply and las files
    find $data_folder/* -type d -exec rm -rf {} + # delete all the folders in the input folder
    echo "Removed all the files and folders except the ply and las files in the input folder"
fi

# move the output of the first step to the input folder of the second step
mkdir -p $data_folder/segmented_point_clouds

# move all .segmented.ply files to the segmented_point_clouds folder if they are in the input folder
find $data_folder/ -type f -name '*.ply' -exec mv {} $data_folder/segmented_point_clouds/ \;

# do the tiling and tile index generation
echo "Tiling and tile index generation"
python nibio_preprocessing/tiling.py \
-i $data_folder/segmented_point_clouds/ \
-o $data_folder/segmented_point_clouds/tiled \
--tile_size 5

# TODO: remove tiles which not dense enough

# iterate over all the directories in the tiled folder
for d in $data_folder/segmented_point_clouds/tiled/*/; do
    for f in $d/*.ply; do
        echo "Processing $f file..."
        python fsct/run.py \
        --point-cloud $f \
        --batch_size 10 \
        --odir $d \
        --verbose \
        --tile-index $d/tile_index.dat \
        --buffer 0.5
    done
done 


# # # iterate over all files in the input folder and do sematic segmentation
# echo  "Starting semantic segmentation"
# for file in $data_folder/*.ply; do
#     # python fsct/run.py --point-cloud $file --batch_size 5 --odir $data_folder --model ./fsct/model/model.pth
#     python fsct/run.py --point-cloud $file --batch_size 5 --odir $data_folder --verbose
# done

# # move the output of the first step to the input folder of the second step
# mkdir -p $data_folder/segmented_point_clouds

# # move all .segmented.ply files to the segmented_point_clouds folder if they are in the input folder
# find $data_folder/ -type f -name '*.segmented.ply' -exec mv {} $data_folder/segmented_point_clouds/ \;

# # do the tiling and tile index generation
# echo "Tiling and tile index generation"
# python nibio_preprocessing/tiling.py -i $data_folder/segmented_point_clouds/ -o $data_folder/segmented_point_clouds/tiled

# # create folder for the output of the second step

# mkdir -p $data_folder/instance_segmented_point_clouds

# echo "done  with the first step"
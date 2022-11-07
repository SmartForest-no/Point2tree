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

# check if conda is activated if not activate it
# if [ -z "$CONDA_DEFAULT_ENV" ]; then
#     echo "Conda is not activated. Activating it"
#     conda activate pdal-env-1
# else
#     echo "Conda is activated"
# fi

# read input folder as a command line argument
# show a message to provide the input folder
data_folder=$1

# if no input folder is provided, case a message and exit
if [ -z "$data_folder" ]
then
    echo "No input folder provided, please provide the input folder as a command line argument"
    exit 1
fi

# check there are las files in the input folder
count=`ls -1 $data_folder/*.las 2>/dev/null | wc -l`

if [ $count != 0 ]; then
    echo "$count las files found in the input folder"
else
    echo "No las files found in the input folder. All files in the input folder should have .las extension."
    exit 1
fi 

#TODO: do the point cloud filtering using nibio_preprocessing/filter_point_cloud.py
echo "Doing the point cloud filtering to density of 150 points per cubic meter"
python nibio_preprocessing/point_cloud_filter.py --dir $data_folder --density 150 --verbose --in_place

# covert all the files to ply format using nibio_preprocessing/convert_files_in_folder.py
echo "Converting all the files to ply format"
python nibio_preprocessing/convert_files_in_folder.py --input_folder $data_folder --output_folder $data_folder --out_file_type ply

# clear input folder if CLEAR_INPUT_FOLDER is set to 1
if [ $CLEAR_INPUT_FOLDER -eq 1 ]
then
    # delete all the files and folders except the ply files in the input folder
    echo "Clearing input folder"
    find $data_folder/ -type f ! -name '*.ply' ! -name '*.las' -delete # delete all the files except the ply and las files
    find $data_folder/* -type d -exec rm -rf {} + # delete all the folders in the input folder
fi

# # iterate over all files in the input folder and do sematic segmentation
echo  "Starting semantic segmentation"
for file in $data_folder/*.ply; do
    # python fsct/run.py --point-cloud $file --batch_size 5 --odir $data_folder --model ./fsct/model/model.pth
    python fsct/run.py --point-cloud $file --batch_size 5 --odir $data_folder --verbose
done

# move the output of the first step to the input folder of the second step
mkdir -p $data_folder/segmented_point_clouds

# move all .segmented.ply files to the segmented_point_clouds folder if they are in the input folder
find $data_folder/ -type f -name '*.segmented.ply' -exec mv {} $data_folder/segmented_point_clouds/ \;

# do the tiling and tile index generation
echo "Tiling and tile index generation"
python nibio_preprocessing/tiling.py -i $data_folder/segmented_point_clouds/ -o $data_folder/segmented_point_clouds/tiled

# create folder for the output of the second step

mkdir -p $data_folder/instance_segmented_point_clouds

# Do the instances and iterate over all the segmented point clouds
for segmented_point_cloud in $data_folder/segmented_point_clouds/*.segmented.ply; do
    # get the name of the segmented point cloud
    segmented_point_cloud_name=$(basename $segmented_point_cloud)
    # get the name of the segmented point cloud without the extension
    segmented_point_cloud_name_no_ext="${segmented_point_cloud_name%.*}"
    # create a directory for the instance segmented point clouds
    mkdir -p $data_folder/instance_segmented_point_clouds/$segmented_point_cloud_name_no_ext
    # iterate over all the tiles of the segmented point cloud
    for tile in $data_folder/segmented_point_clouds/tiled/$segmented_point_cloud_name_no_ext/*.ply; do
        # get the name of the tile
        tile_name=$(basename $tile)
        # get the name of the tile without the extension
        tile_name_no_ext="${tile_name%.*}"
        echo "Processing $tile"
        # show the output folder
        echo "Output folder: $data_folder/instance_segmented_point_clouds/$segmented_point_cloud_name_no_ext/$tile_name_no_ext"
        python3 fsct/points2trees.py \
        -t $tile \
        --tindex $data_folder/segmented_point_clouds/tiled/$segmented_point_cloud_name_no_ext/tile_index.dat \
        -o $data_folder/instance_segmented_point_clouds/$segmented_point_cloud_name_no_ext/$tile_name_no_ext \
        --n-tiles $N_TILES \
        --slice-thickness $SLICE_THICKNESS \
        --find-stems-height $FIND_STEMS_HEIGHT \
        --find-stems-thickness $FIND_STEMS_THICKNESS \
        --pandarallel --verbose \
        --add-leaves \
        --add-leaves-voxel-length $ADD_LEAVES_VOXEL_LENGTH \
        --graph-maximum-cumulative-gap $GRAPH_MAXIMUM_CUMULATIVE_GAP \
        --save-diameter-class \
        --ignore-missing-tiles \
        --find-stems-min-points $FIND_STEMS_MIN_POINTS
    done
done

# do merging of the instance segmented point clouds
for instance_segmented_point_cloud in $data_folder/instance_segmented_point_clouds/*; do
    python nibio_preprocessing/merging_and_labeling.py \
    --data_folder $instance_segmented_point_cloud \
    --output_file $instance_segmented_point_cloud/output_instance_segmented.ply
done

# # create the results folder
mkdir -p $data_folder/results

# # create the input data folder
mkdir -p $data_folder/results/input_data

# # move input data (ply and las) to the input data folder
find $data_folder/ -maxdepth 1 -type f -name '*.ply' -exec mv {} $data_folder/results/input_data/ \;
find $data_folder/ -maxdepth 1 -type f -name '*.las' -exec mv {} $data_folder/results/input_data/ \;

# # create the segmented point clouds folder
mkdir -p $data_folder/results/segmented_point_clouds

# move segmented point clouds to the segmented point clouds folder
find $data_folder/segmented_point_clouds/ -maxdepth 1 -type f -name '*.ply' -exec mv {} $data_folder/results/segmented_point_clouds/ \;

# # create the instance segmented point clouds folder
mkdir -p $data_folder/results/instance_segmented_point_clouds

# iterate over all the instance segmented point clouds 
# move instance segmented point clouds to the instance segmented point clouds folder and rename them
for instance_segmented_point_cloud in $data_folder/instance_segmented_point_clouds/*; do
    # get the name of the instance segmented point cloud
    instance_segmented_point_cloud_name=$(basename $instance_segmented_point_cloud)
    # get the name of the instance segmented point cloud without the extension
    instance_segmented_point_cloud_name_no_ext="${instance_segmented_point_cloud_name%.*}"
    # move the instance segmented point cloud to the instance segmented point clouds folder
    find $instance_segmented_point_cloud/ -maxdepth 1 -type f -name '*.ply' -exec mv {} $data_folder/results/instance_segmented_point_clouds/$instance_segmented_point_cloud_name_no_ext.ply \;
    # map the instance segmented point cloud to las file
    pdal translate \
    $data_folder/results/instance_segmented_point_clouds/$instance_segmented_point_cloud_name_no_ext.ply \
    $data_folder/results/instance_segmented_point_clouds/$instance_segmented_point_cloud_name_no_ext.las \
    --writers.las.dataformat_id=3 \
    --writers.las.scale_x=0.01 \
    --writers.las.scale_y=0.01 \
    --writers.las.scale_z=0.01 \
    --writers.las.extra_dims=all
done

echo "Done"
echo "Results are in $data_folder/results"





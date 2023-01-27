#!/bin/bash

############################ parameters #################################################
# General parameters
CLEAR_INPUT_FOLDER=1  # 1: clear input folder, 0: not clear input folder
CONDA_ENV="pdal-env-1" # conda environment for running the pipeline

# Tiling parameters
data_folder="" # path to the folder containing the data
N_TILES=3
SLICE_THICKNESS=0.5
FIND_STEMS_HEIGHT=1.5
FIND_STEMS_THICKNESS=0.5
GRAPH_MAXIMUM_CUMULATIVE_GAP=3
ADD_LEAVES_VOXEL_LENGTH=0.5
FIND_STEMS_MIN_POINTS=50
GRAPH_EDGE_LENGTH=1.0
ADD_LEAVES_EDGE_LENGTH=1.0

############################# end of parameters declaration ############################

# extract tiling parameters as command line arguments with the same default values
while getopts "d:n:s:h:t:g:l:m:o:p:" opt; do
  case $opt in
    d) data_folder="$OPTARG"
    ;;
    n) N_TILES="$OPTARG"
    ;;
    s) SLICE_THICKNESS="$OPTARG"
    ;;
    h) FIND_STEMS_HEIGHT="$OPTARG"
    ;;
    t) FIND_STEMS_THICKNESS="$OPTARG"
    ;;
    g) GRAPH_MAXIMUM_CUMULATIVE_GAP="$OPTARG"
    ;;
    l) ADD_LEAVES_VOXEL_LENGTH="$OPTARG"
    ;;
    m) FIND_STEMS_MIN_POINTS="$OPTARG"
    ;;
    o) GRAPH_EDGE_LENGTH="$OPTARG"
    ;;
    p) ADD_LEAVES_EDGE_LENGTH="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# print the letters to choose from in getopts
echo "      The list of letters for the parameters:"
echo "d: data_folder"
echo "n: N_TILES"
echo "s: SLICE_THICKNESS"
echo "h: FIND_STEMS_HEIGHT"
echo "t: FIND_STEMS_THICKNESS"
echo "g: GRAPH_MAXIMUM_CUMULATIVE_GAP"
echo "l: ADD_LEAVES_VOXEL_LENGTH"
echo "m: FIND_STEMS_MIN_POINTS"
echo "o: GRAPH_EDGE_LENGTH"
echo "p: ADD_LEAVES_EDGE_LENGTH"

echo " "
# print values of the parameters 
echo "      The values of the parameters:"
echo "data_folder: $data_folder"
echo "N_TILES: $N_TILES"
echo "SLICE_THICKNESS: $SLICE_THICKNESS"
echo "FIND_STEMS_HEIGHT: $FIND_STEMS_HEIGHT"
echo "FIND_STEMS_THICKNESS: $FIND_STEMS_THICKNESS"
echo "GRAPH_MAXIMUM_CUMULATIVE_GAP: $GRAPH_MAXIMUM_CUMULATIVE_GAP"
echo "ADD_LEAVES_VOXEL_LENGTH: $ADD_LEAVES_VOXEL_LENGTH"
echo "FIND_STEMS_MIN_POINTS: $FIND_STEMS_MIN_POINTS"
echo "GRAPH_EDGE_LENGTH: $GRAPH_EDGE_LENGTH"
echo "ADD_LEAVES_EDGE_LENGTH: $ADD_LEAVES_EDGE_LENGTH"

# exit 0

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

# if no input folder is provided, case a message and exit
if [ -z "$data_folder" ]
then
    echo " "
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
--tile_size 10

# remove small tiles using nibio_preprocessing/remove_small_tiles.py
for d in $data_folder/segmented_point_clouds/tiled/*; do
    echo "Removing small tiles from $d"
    python nibio_preprocessing/remove_small_tiles.py \
    --dir $d \
    --tile_index $d/tile_index.dat \
    --min_density 75 \
    --verbose
done

# iterate over all the directories in the tiled folder
for d in $data_folder/segmented_point_clouds/tiled/*/; do
    for f in $d/*.ply; do
        echo "Processing $f file..."
        python fsct/run.py \
        --model /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/fsct/model/model.pth \
        --point-cloud $f \
        --batch_size 10 \
        --odir $d \
        --verbose \
        # --tile-index $d/tile_index.dat \
        # --buffer 2
    done
done

# remove all the files in the tiled subfolders except the *segmented.ply and tile_index.dat files
find $data_folder/segmented_point_clouds/tiled/*/ -type f ! -name '*segmented.ply' ! -name 'tile_index.dat' -delete # delete all the files except the segmented.ply files
# delete all the folders in the tiled subfolders
find $data_folder/segmented_point_clouds/tiled/*/* -type d -exec rm -rf {} +

# # merge the segmented point clouds
echo "Merging the segmented point clouds"
# iterate over all the directories in the tiled folder
for d in $data_folder/segmented_point_clouds/tiled/*/; do
    # get a base name of the directory
    base_name=$(basename $d)
    # create a name for the merged file
    merged_file_name=$data_folder/segmented_point_clouds/$base_name.segmented.ply
    python nibio_preprocessing/merging_and_labeling.py \
    --data_folder $d \
    --output_file $merged_file_name \
    --only_merging
done

# rename all the segmented.ply files to .ply in the tiled subfolders
for file in $data_folder/segmented_point_clouds/tiled/*/*; do
    # skip if the file is not a ply file
    if [[ $file != *.ply ]]; then
        continue
    fi
    mv -- "$file" "${file%.segmented.ply}.ply"
done

# rename all the folder in the tiled subfolders to .segmented suffix
for d in $data_folder/segmented_point_clouds/tiled/*; do
    echo "Renaming $d to ${d%.segmented}"
    # mv "$d" "${d%.segmented}"
    mv $d{,.segmented}
done


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
        --find-stems-min-points $FIND_STEMS_MIN_POINTS \
        --graph-edge-length $GRAPH_EDGE_LENGTH \
        --add-leaves-edge-length $ADD_LEAVES_EDGE_LENGTH 
    done
done

# do merging of the instance segmented point clouds
for instance_segmented_point_cloud in $data_folder/instance_segmented_point_clouds/*; do
    python nibio_preprocessing/merging_and_labeling.py \
    --data_folder $instance_segmented_point_cloud \
    --output_file $instance_segmented_point_cloud/output_instance_segmented.ply
done

# create the results folder
mkdir -p $data_folder/results

# # create the input data folder
mkdir -p $data_folder/results/input_data

# # move input data (ply and las) to the input data folder
find $data_folder/ -maxdepth 1 -type f -name '*.ply' -exec mv {} $data_folder/results/input_data/ \;
find $data_folder/ -maxdepth 1 -type f -name '*.las' -exec mv {} $data_folder/results/input_data/ \;

# # create the segmented point clouds folder
mkdir -p $data_folder/results/segmented_point_clouds

# move segmented point clouds to the segmented point clouds folder
find $data_folder/segmented_point_clouds/ -maxdepth 1 -type f -name '*segmented.ply' -exec mv {} $data_folder/results/segmented_point_clouds/ \;

# # create the instance segmented point clouds folder
mkdir -p $data_folder/results/instance_segmented_point_clouds

# iterate over all the instance segmented point clouds
# move instance segmented point clouds to the instance segmented point clouds folder and rename them
for instance_segmented_point_cloud in $data_folder/instance_segmented_point_clouds/*; do
    # get the name of the instance segmented point cloud
    instance_segmented_point_cloud_name=$(basename $instance_segmented_point_cloud)
    # get the name of the instance segmented point cloud without the extension and add the suffix instance_segmented
    instance_segmented_point_cloud_name_no_ext="${instance_segmented_point_cloud_name%.*}.instance_segmented"
    # move the instance segmented point cloud to the instance segmented point clouds folder
    find $instance_segmented_point_cloud/ -maxdepth 1 -type f -name '*.ply' -exec mv {} $data_folder/results/instance_segmented_point_clouds/$instance_segmented_point_cloud_name_no_ext.ply \;
    # map the instance segmented point cloud to las file
    pdal translate \
    $data_folder/results/instance_segmented_point_clouds/$instance_segmented_point_cloud_name_no_ext.ply \
    $data_folder/results/instance_segmented_point_clouds/$instance_segmented_point_cloud_name_no_ext.las \
    --writers.las.dataformat_id=3 \
    --writers.las.extra_dims=all
done

 # change the names of the segmented files to *.segmented.las
for segmented_point_cloud_in_ply in $data_folder/results/segmented_point_clouds/*; do
    # get the prefix of the point clouds
    SEGMENTED_POINT_CLOUDS_PREFIX="segmented."
    # get the ending of the point clouds
    SEGMENTED_POINT_CLOUDS_EXTENSION="ply"
    # get the name of the ply point cloud
    segmented_point_cloud_in_ply_name=$(basename $segmented_point_cloud_in_ply)
    # got the name of the las file without the starting prefix and the .ply extension
    segmented_point_cloud_in_las_name_no_prefix_no_extension=${segmented_point_cloud_in_ply_name#$SEGMENTED_POINT_CLOUDS_PREFIX}
    segmented_point_cloud_in_las_name_no_extension=${segmented_point_cloud_in_las_name_no_prefix_no_extension%.$SEGMENTED_POINT_CLOUDS_EXTENSION}
    # convert it to las and move it to the segmented point clouds folder
    pdal translate \
    $segmented_point_cloud_in_ply \
    $data_folder/results/segmented_point_clouds/$segmented_point_cloud_in_las_name_no_extension.las \
    --writers.las.dataformat_id=3 \
    --writers.las.extra_dims=all
done

python nibio_preprocessing/add_ground_to_inst_seg_folders.py --sem_seg_folder sample_playground/results/segmented_point_clouds/ --inst_seg_folder sample_playground/results/instance_segmented_point_clouds/ --output_folder sample_playground/instance_seg_with_ground --verbose

# create the instance segmented point clouds with ground folder
mkdir -p $data_folder/results/instance_segmented_point_clouds_with_ground

# to add the ground to the instance segmented point clouds
python nibio_preprocessing/add_ground_to_inst_seg_folders.py \
--sem_seg_folder $data_folder/results/segmented_point_clouds/ \
--inst_seg_folder $data_folder/results/instance_segmented_point_clouds/ \
--output_folder $data_folder/results/instance_segmented_point_clouds_with_ground \
--verbose

echo " "
echo "Done"
# print path to the results folder and the subfolders
echo "Results can be found here: $data_folder/results"
echo "Results containing the input point clouds can be found here:  $data_folder/results/input_data"
echo "Results containing the segmented point clouds can be found here:  $data_folder/results/segmented_point_clouds"
echo "Results containing the instance segmented point clouds can be found here:  $data_folder/results/instance_segmented_point_clouds"
echo "Results containing the instance segmented point clouds with ground can be found here:  $data_folder/results/instance_segmented_point_clouds_with_ground"


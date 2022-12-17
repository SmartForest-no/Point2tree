#!/bin/bash

############################ parameters #################################################
# General parameters
CLEAR_INPUT_FOLDER=1  # 1: clear input folder, 0: not clear input folder
CONDA_ENV="pdal-env-1" # conda environment for running the pipeline

# Tiling parameters
data_folder="" # path to the folder containing the data
remove_small_tiles=0 # 1: remove small tiles, 0: not remove small tiles

############################# end of parameters declaration ############################

# extract tiling parameters as command line arguments with the same default values

# add remove_small_tiles parameter
while getopts "d:z:" opt; do
  case $opt in
    d) data_folder="$OPTARG"
    ;;
    z) remove_small_tiles="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# print the letters to choose from in getopts
echo "      The list of letters for the parameters:"
echo "d: data_folder"

# print values of the parameters 
echo "      The values of the parameters:"
echo "data_folder: $data_folder"
echo "remove_small_tiles: $remove_small_tiles"

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

# make it conditional bassed remove_small_tiles parameter
if  [ $remove_small_tiles -eq 1 ]
then
    # iterate over all the directories in the tiled folder
    for d in $data_folder/segmented_point_clouds/tiled/*; do
        echo "Removing small tiles from $d"
        python nibio_preprocessing/remove_small_tiles.py \
        --dir $d \
        --tile_index $d/tile_index.dat \
        --min_density 75 \
        --verbose
    done
fi

# iterate over all the directories in the tiled folder
for d in $data_folder/segmented_point_clouds/tiled/*/; do
    for f in $d/*.ply; do
        echo "Processing $f file..."
        python sean_sem_seg/run_single_file.py \
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

echo "Semantic segmentation done."

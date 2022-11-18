#!/bin/bash

# run run_all_command_lines.sh with the following arguments:

data_folder="/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground" 
N_TILES=3

SLICE_THICKNESS=0.382368454442735
FIND_STEMS_HEIGHT=1.8948172056774
FIND_STEMS_THICKNESS=0.9980435744231868
GRAPH_MAXIMUM_CUMULATIVE_GAP=13.841583930676254
ADD_LEAVES_VOXEL_LENGTH=0.19332721135500391
FIND_STEMS_MIN_POINTS=495
GRAPH_EDGE_LENGTH=0.5652008887940575
ADD_LEAVES_EDGE_LENGTH=0.5622733957401558

# get test data from the following link:
bash ./bash_helper_scripts/get_terrestial_sem_seg_test.sh

# run run_all_command_lines.sh with the following arguments:
./run_all_command_line.sh -d $data_folder \
-n $N_TILES \
-s $SLICE_THICKNESS \
-h $FIND_STEMS_HEIGHT \
-t $FIND_STEMS_THICKNESS \
-g $GRAPH_MAXIMUM_CUMULATIVE_GAP \
-l $ADD_LEAVES_VOXEL_LENGTH \
-m $FIND_STEMS_MIN_POINTS \
-o $GRAPH_EDGE_LENGTH \
-p $ADD_LEAVES_EDGE_LENGTH
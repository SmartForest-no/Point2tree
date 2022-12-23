#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground
# clean the folder
rm -rf $TARGET_FOLDER/*

cp /home/nibio/mutable-outside-world/data/corrected/test/*.las $TARGET_FOLDER

# remove Plot71_tile_-25_-25.las from the folder
# this file breaks the test pipeline
rm $TARGET_FOLDER/Plot71_tile_-25_-25.las



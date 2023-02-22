#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek_optimized
# clean the folder
rm -rf $TARGET_FOLDER/*

cp -r /home/nibio/mutable-outside-world/data/austrian_data_after_seg/* $TARGET_FOLDER


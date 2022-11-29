#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek
# clean the folder
rm -rf $TARGET_FOLDER/*

cp /home/nibio/mutable-outside-world/data/corrected/validation/*.las $TARGET_FOLDER


#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek
# clean the folder
rm -rf $TARGET_FOLDER/*

cp /home/nibio/mutable-outside-world/data/austrian_example_stefano/p2_instance.las $TARGET_FOLDER


#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek
# clean the folder
rm -rf $TARGET_FOLDER/*

cp /home/nibio/mutable-outside-world/data/terestial_data_for_training_sem_seg_model/test/plot73_tile_-25_-25.las $TARGET_FOLDER


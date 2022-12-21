#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek
# clean the folder
rm -rf $TARGET_FOLDER/*

cp -r /home/nibio/mutable-outside-world/data/terestial_data_for_training_sem_seg_model/validation/burum_2_tile_-100_0.las $TARGET_FOLDER
cp -r /home/nibio/mutable-outside-world/data/terestial_data_for_training_sem_seg_model/validation/plot82_tile_-25_0.las $TARGET_FOLDER
cp -r /home/nibio/mutable-outside-world/data/terestial_data_for_training_sem_seg_model/validation/plot72_tile_-25_0.las $TARGET_FOLDER


#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground
# clean the folder
rm -rf $TARGET_FOLDER/*

cp -r /home/nibio/mutable-outside-world/data/terestial_data_for_training_sem_seg_model/validation_after_seg/* $TARGET_FOLDER


#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek
# clean the folder
rm -rf $TARGET_FOLDER/*

cp -r /home/nibio/mutable-outside-world/data/corrected/validation/burum2_tile_-100_0.las $TARGET_FOLDER
cp -r /home/nibio/mutable-outside-world/data/corrected/validation/Plot89_tile_-25_0.las $TARGET_FOLDER
cp -r /home/nibio/mutable-outside-world/data/corrected/validation/plot82_tile_-25_0.las $TARGET_FOLDER
# cp -r /home/nibio/mutable-outside-world/data/corrected/validation/Plot104_tile_-25_0.las $TARGET_FOLDER



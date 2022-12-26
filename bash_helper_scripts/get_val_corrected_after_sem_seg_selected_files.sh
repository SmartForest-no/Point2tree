#!/bin/bash

TARGET_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground
# clean the folder
rm -rf $TARGET_FOLDER/*

mkdir $TARGET_FOLDER/segmented_point_clouds
mkdir $TARGET_FOLDER/instance_segmented_point_clouds

cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/burum2_tile_-100_0.las $TARGET_FOLDER
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/Plot89_tile_-25_0.las $TARGET_FOLDER
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/plot82_tile_-25_0.las $TARGET_FOLDER
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/Plot104_tile_-25_0.las $TARGET_FOLDER

cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/burum2_tile_-100_0.segmented.ply $TARGET_FOLDER/segmented_point_clouds
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/Plot89_tile_-25_0.segmented.ply $TARGET_FOLDER/segmented_point_clouds
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/plot82_tile_-25_0.segmented.ply $TARGET_FOLDER/segmented_point_clouds
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/Plot104_tile_-25_0.segmented.ply $TARGET_FOLDER/segmented_point_clouds

cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/tiled/burum2_tile_-100_0.segmented $TARGET_FOLDER/segmented_point_clouds/tiled
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/tiled/Plot89_tile_-25_0.segmented $TARGET_FOLDER/segmented_point_clouds/tiled
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/tiled/plot82_tile_-25_0.segmented $TARGET_FOLDER/segmented_point_clouds/tiled
cp -r /home/nibio/mutable-outside-world/data/corrected/validation_after_seg/segmented_point_clouds/tiled/Plot104_tile_-25_0.segmented $TARGET_FOLDER/segmented_point_clouds/tiled




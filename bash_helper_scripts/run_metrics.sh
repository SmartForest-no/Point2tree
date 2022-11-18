#!/bin/bash

python metrics/instance_segmentation_metrics_in_folder.py \
--gt_las_folder_path /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/old_sample_playground/results/input_data/ \
--target_las_folder_path /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/old_sample_playground/results/instance_segmented_point_clouds \
--remove_ground \
--output_folder_path /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/old_sample_playground/results/metrics \
--verbose 

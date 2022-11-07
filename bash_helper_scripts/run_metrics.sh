#!/bin/bash

python nibio_postprocessing/get_instances_side_by_side.py \
--input_folder /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds/ \
--output_folder /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds/output \
--instance_label 'instance_nr' \
--verbose \
# --merge

python nibio_postprocessing/get_instances_side_by_side.py \
--input_folder /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds_with_ground/ \
--output_folder /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds_with_ground/output \
--instance_label 'instance_nr' \
--verbose \
# --merge

python nibio_postprocessing/get_instances_side_by_side.py \
--input_folder /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/input_data/ \
--output_folder /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/input_data/output \
--instance_label 'treeID' \
--verbose \
# --merge


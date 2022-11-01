# remove all the files in the source folder
SOURCE_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground
rm -rf $SOURCE_FOLDER/*

cp /home/nibio/mutable-outside-world/data/strange_shape_cloud_for_pipeline_test/2022-08-05_11-03-31_9pct_time_scan.laz \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground

# change name of the file to first.laz 
mv /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/2022-08-05_11-03-31_9pct_time_scan.laz \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/first.laz

# # copy first.laz to second.laz
cp /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/first.laz \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/second.laz
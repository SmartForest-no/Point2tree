SOURCE_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground
rm -rf $SOURCE_FOLDER/*

cp /home/nibio/mutable-outside-world/data/small_file_pipeline_test/small_file_pipeline_test.las \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground

# change name of the file to first.laz 
mv /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/small_file_pipeline_test.las \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/first.las

# copy first.laz to second.laz
cp /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/first.las \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/second.las
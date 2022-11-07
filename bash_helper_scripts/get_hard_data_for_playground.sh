# remove all the files in the source folder
SOURCE_FOLDER=/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground
rm -rf $SOURCE_FOLDER/*

cp /home/nibio/mutable-outside-world/data/raw_for_pipeline_test/Plot69_2022-06-15_09-08-53_9pct_time.las \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground

# change name of the file to first.laz 
mv /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/Plot69_2022-06-15_09-08-53_9pct_time.las \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/first.las

# copy first.laz to second.laz
cp /home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/first.las \
/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/second.las
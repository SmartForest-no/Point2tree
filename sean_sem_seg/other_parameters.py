# Don't change these unless you really understand what you are doing with them/are learning how the code works.
# These have been tuned to work on most high resolution forest point clouds without changing them, but you may be able
# to tune these better for your particular data. Almost everything here is a trade-off between different situations, so
# optimisation is not straight-forward.

other_parameters = dict(
    model_filename="/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/fsct/model/model.pth",
    box_dimensions=[6, 6, 8],  # Dimensions of the sliding box used for semantic segmentation.
    box_overlap=[0.5, 0.5, 0.5],  # Overlap of the sliding box used for semantic segmentation.
    min_points_per_box=1000,  # Minimum number of points for input to the model. Too few points and it becomes near impossible to accurately label them (though assuming vegetation class is the safest bet here).
    max_points_per_box=20000,  # Maximum number of points for input to the model. The model may tolerate higher numbers if you decrease the batch size accordingly (to fit on the GPU), but this is not tested.
    num_classes = 4,  # Number of classes in the model. Don't change this unless you are changing the model.
    noise_class=0,  # Don't change
    terrain_class=1,  # Don't change
    vegetation_class=2,  # Don't change
    cwd_class=3,  # Don't change
    stem_class=4,  # Don't change
    branch_class=5,  # Don't change
    low_vegetation_class=6,  # Don't change
    grid_resolution=0.5,  # Resolution of the DTM.
    vegetation_coverage_resolution=0.2,
    num_neighbours=5,
    sorting_search_angle=20,
    sorting_search_radius=1,
    sorting_angle_tolerance=90,
    max_search_radius=3,
    max_search_angle=30,
    min_cluster_size=30,  # Used for HDBSCAN clustering step. Recommend not changing for general use.
    cleaned_measurement_radius=0.2,  # During cleaning, this w
    subsample=1,  # Generally leave this on, but you can turn off subsampling.
    subsampling_min_spacing=0.01,  # The point cloud will be subsampled such that the closest any 2 points can be is 0.01 m.
    minimum_CCI=0.3,  # Minimum valid Circuferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
    min_tree_cyls=10,  # Deletes any trees with fewer than 10 cylinders (before the cylinder interpolation step).
    low_resolution_point_cloud_hack_mode=0,
    seed=42,
)  # Very ugly hack that can sometimes be useful on point clouds which are on the borderline of having not enough points to be functional with FSCT. Set to a positive integer. Point cloud will be copied this many times (with noise added) to artificially increase point density giving the segmentation model more points.

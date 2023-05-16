import argparse
import json
import pdal
import laspy
import os
import shutil
from other_parameters import other_parameters

from helpers.reduce_labels_values_in_las import ReduceLabelsValuesInLas

from seed_everything import seed_everything
seed_everything(other_parameters['seed'])

from run_tools import FSCT

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the FSCT pipeline of Sean semantic seg.")
    parser.add_argument("--point-cloud", "-p", default="", type=str, help="path to point cloud", required=True)
    parser.add_argument("--model", "-m", default="", type=str, help="path to model")
    parser.add_argument("--batch_size", "-b", default=5, type=int, help="batch size")
    parser.add_argument("--odir", "-o", default=".", type=str, help="output directory")
    parser.add_argument("--verbose", help="Print more information.", action="store_true")

    args = parser.parse_args()

    # check if output directory exists
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    
    # check if the point cloud is a las file
    if not args.point_cloud.endswith(".las"):
        print("Converting point cloud to las format.")
        # convert to las and name it as the original file name
        las_file = os.path.join(args.odir, os.path.basename(args.point_cloud).split(".")[0] + ".las")
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [
                args.point_cloud,
                {
                    "type": "writers.las",
                    "filename": las_file
                }
            ]
        }))
        pipeline.execute()
        args.point_cloud = las_file


    parameters = dict(
        point_cloud_filename=args.point_cloud,
        # Adjust if needed
        plot_centre=None,  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
        # Circular Plot options - Leave at 0 if not using.
        plot_radius=0,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
        plot_radius_buffer=0,  # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".
        # Set these appropriately for your hardware.
        batch_size=args.batch_size,  # You will get CUDA errors if this is too high, as you will run out of VRAM. This won't be an issue if running on CPU only. Must be >= 2.
        num_cpu_cores=0,  # Number of CPU cores you want to use. If you run out of RAM, lower this. 0 means ALL cores.
        use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
        # Optional settings - Generally leave as they are.
        slice_thickness=0.15,  # If your point cloud resolution is a bit low (and only if the stem segmentation is still reasonably accurate), try increasing this to 0.2.
        # If your point cloud is really dense, you may get away with 0.1.
        slice_increment=0.05,  # The smaller this is, the better your results will be, however, this increases the run time.
        sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
        # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.
        height_percentile=100,  # If the data contains noise above the canopy, you may wish to set this to the 98th percentile of height, otherwise leave it at 100.
        tree_base_cutoff_height=5,  # A tree must have a cylinder measurement below this height above the DTM to be kept. This filters unsorted branches from being called individual trees.
        generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
        # If you activate "tree aware plot cropping mode", this function will use it.
        ground_veg_cutoff_height=3,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
        veg_sorting_range=1.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
        stem_sorting_range=1,  # Stem points can be, at most, this far away from a cylinder in 3D to be matched to a particular tree.
        taper_measurement_height_min=0,  # Lowest height to measure diameter for taper output.
        taper_measurement_height_max=30,  # Highest height to measure diameter for taper output.
        taper_measurement_height_increment=0.2,  # diameter measurement increment.
        taper_slice_thickness=0.4,  # Cylinder measurements within +/- 0.5*taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used.
        delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished.
        # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
        minimise_output_size_mode=0,  # Will delete a number of non-essential outputs to reduce storage use.
        grid_resolution=0.1,  # Resolution of the grid used for the DTM.
    )

    parameters.update(other_parameters)
    parameters["model_filename"] = args.model

    # print path to model
    print("Using model: {}".format(parameters["model_filename"]))

    FSCT(
        parameters=parameters,
        # Set below to 0 or 1 (or True/False). Each step requires the previous step to have been run already.
        # For standard use, just leave them all set to 1 except "clean_up_files".
        preprocess=True,  # Preparation for semantic segmentation.
        segmentation=True,  # Deep learning based semantic segmentation of the point cloud.
        postprocessing=True,  # Creates the DTM and applies some simple rules to clean up the segmented point cloud.
        measure_plot=False,  # The bulk of the plot measurement happens here.
        make_report=False,  # Generates a plot report, plot map, and some other figures.
        clean_up_files=False,
    )  # Optionally deletes most of the large point cloud outputs to minimise storage requirements.

    # copy the output "segmented_cleaned.las" to the output directory
    
    # results_dir_name = args.point_cloud.split('.')[0] + '_FSCT_output'
    dir_core_name = os.path.dirname(args.point_cloud)
    file_name = os.path.basename(args.point_cloud).split('.')[0]
    results_dir_name = os.path.join(dir_core_name, file_name + '_FSCT_output')

    print("Copying results to output directory.")
    shutil.copy(os.path.join(results_dir_name, "segmented_cleaned.las"), args.odir)

    print("Doing reduction.")

    # reduce label value by 1
    ReduceLabelsValuesInLas(
        las_file_path = os.path.join(args.odir, "segmented_cleaned.las"),
        label_name="label",
        verbose=args.verbose
        ).main()

    # translate segmented_cleaned.las to segmented_cleaned.ply using pdal
    # create a pipeline for the translation
    pipeline = [
        {
            "type": "readers.las",
            "nosrs": True,
            "filename": os.path.join(args.odir, "segmented_cleaned.las"),
        },
        {
            "type": "writers.ply",
            "filename": os.path.join(args.odir, "segmented_cleaned.ply")
        },
    ]
    # create a pipeline manager
    pl = pdal.Pipeline(json.dumps(pipeline))
    # execute the pipeline
    pl.execute()
    # rename the output file from "segmented_cleaned.las" to args.point_cloud.segmented.las
    os.rename(os.path.join(args.odir, "segmented_cleaned.ply"), os.path.join(args.odir, args.point_cloud.split('/')[-1].split('.')[0] + ".segmented.ply"))


import subprocess

import wandb
# local imports
from metrics.instance_segmentation_metrics_in_folder import \
    InstanceSegmentationMetricsInFolder

# wandb.login()

# wandb.init(project="instance_segmentation_classic", entity="smart_forest")

# define a class to run the command with arguments
class RunCommand:
    def __init__(self, cmd, args):
        self.cmd = cmd
        self.args = args

    def __call__(self):
        print("Running command: " + self.cmd + " " + " ".join(self.args))
        subprocess.run([self.cmd, *self.args])

def main(
    n_tiles, 
    slice_thickness, 
    find_stems_height, 
    find_stems_thickness, 
    graph_maximum_cumulative_gap, 
    add_leaves_voxel_length, 
    find_stems_min_points, 
    graph_edge_length, 
    add_leaves_edge_length
    ):

    # initialize the sweep
    run = wandb.init(project="paper-sweep-nibio-model", entity="smart_forest")

    # get files for the sweep
    print("Getting files for the sweep")
    cmd = "/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/bash_helper_scripts/get_terrestial_sem_seg_validation.sh"
    subprocess.run([cmd], shell=True)

    # define the arguments for all the parameters from the sweep configuration
    print("Defining arguments for all the parameters from the sweep configuration")
    
    # print the arguments
    print("N_TILES: " + str(n_tiles))
    print("SLICE_THICKNESS: " + str(slice_thickness))
    print("FIND_STEMS_HEIGHT: " + str(find_stems_height))
    print("FIND_STEMS_THICKNESS: " + str(find_stems_thickness))
    print("GRAPH_MAXIMUM_CUMULATIVE_GAP: " + str(graph_maximum_cumulative_gap))
    print("ADD_LEAVES_VOXEL_LENGTH: " + str(add_leaves_voxel_length))
    print("FIND_STEMS_MIN_POINTS: " + str(find_stems_min_points))
    print("GRAPH_EDGE_LENGTH: " + str(graph_edge_length))
    print("ADD_LEAVES_EDGE_LENGTH: " + str(add_leaves_edge_length))

    # define the command
    cmd = "./run_all_command_line.sh"
    
    # define the arguments
    args = [
        "-d", "/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground"
        ]

    print("Adding the arguments to the list of arguments")
    args.extend([
        "-n", str(n_tiles),
        "-s", str(slice_thickness),
        "-h", str(find_stems_height),
        "-t", str(find_stems_thickness),
        "-g", str(graph_maximum_cumulative_gap),
        "-l", str(add_leaves_voxel_length),
        "-m", str(find_stems_min_points),
        "-o", str(graph_edge_length),
        "-p", str(add_leaves_edge_length)
    ])

    # run the command with the arguments
    print("Running the command with the arguments")
    RunCommand(cmd, args)()

    # compute the metric
    print("Computing the metric")
    metric = InstanceSegmentationMetricsInFolder(
        gt_las_folder_path = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/input_data',
        target_las_folder_path = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds',
        remove_ground=True,
        verbose=True
    ) 

    f1_score = metric.main()
    print("F1 score: " + str(f1_score))

    # log the metric
    print("Logging the metric")
    wandb.log({"f1_score": f1_score})

if __name__ == "__main__":
    # use argparse to get the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tiles", type=int, default=1)
    parser.add_argument("--slice_thickness", type=float, default=0.1)
    parser.add_argument("--find_stems_height", type=float, default=0.1)
    parser.add_argument("--find_stems_thickness", type=float, default=0.1)
    parser.add_argument("--graph_maximum_cumulative_gap", type=float, default=0.1)
    parser.add_argument("--add_leaves_voxel_length", type=float, default=0.1)
    parser.add_argument("--find_stems_min_points", type=int, default=1)
    parser.add_argument("--graph_edge_length", type=float, default=0.1)
    parser.add_argument("--add_leaves_edge_length", type=float, default=0.1)
    args = parser.parse_args()

    # run the main function
    main(
        args.n_tiles,
        args.slice_thickness,
        args.find_stems_height,
        args.find_stems_thickness,
        args.graph_maximum_cumulative_gap,
        args.add_leaves_voxel_length,
        args.find_stems_min_points,
        args.graph_edge_length,
        args.add_leaves_edge_length
    )




import subprocess
import wandb

# local imports
from metrics.instance_segmentation_metrics_in_folder import InstanceSegmentationMetricsInFolder

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

# define the sweep configuration with the parameters to sweep
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'f1_score'},
    'parameters':
    {
        'N_TILES': {'values': [3]},
        'SLICE_THICKNESS': {'values': [0.25, 0.5, 0.75]},
        'FIND_STEMS_HEIGHT': {'values': [0.5, 0.75, 1.0, 1.5, 2.0]},
        'FIND_STEMS_THICKNESS': {'values': [0.25, 0.5, 0.75]},
        'GRAPH_MAXIMUM_CUMULATIVE_GAP': {'values': [1, 2, 3, 4]},
        'ADD_LEAVES_VOXEL_LENGTH': {'values': [0.25, 0.5, 0.75]},
        'FIND_STEMS_MIN_POINTS': {'values': [10, 20, 30, 50, 100, 150, 200]}
    }
}

def main():
    # initialize the sweep
    run = wandb.init(project="sweep-train", entity="maciej-wielgosz-nibio")

    # get files for the sweep
    print("Getting files for the sweep")
    cmd = "/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/bash_helper_scripts/get_small_data_for_playground.sh"
    subprocess.run([cmd], shell=True)

    # define the arguments for all the parameters from the sweep configuration
    print("Defining arguments for all the parameters from the sweep configuration")
    n_tiles = wandb.config.N_TILES
    slice_thickness = wandb.config.SLICE_THICKNESS
    find_stems_height = wandb.config.FIND_STEMS_HEIGHT
    find_stems_thickness = wandb.config.FIND_STEMS_THICKNESS
    graph_maximum_cumulative_gap = wandb.config.GRAPH_MAXIMUM_CUMULATIVE_GAP
    add_leaves_voxel_length = wandb.config.ADD_LEAVES_VOXEL_LENGTH
    find_stems_min_points = wandb.config.FIND_STEMS_MIN_POINTS

    # print the arguments
    print("N_TILES: " + str(n_tiles))
    print("SLICE_THICKNESS: " + str(slice_thickness))
    print("FIND_STEMS_HEIGHT: " + str(find_stems_height))
    print("FIND_STEMS_THICKNESS: " + str(find_stems_thickness))
    print("GRAPH_MAXIMUM_CUMULATIVE_GAP: " + str(graph_maximum_cumulative_gap))
    print("ADD_LEAVES_VOXEL_LENGTH: " + str(add_leaves_voxel_length))
    print("FIND_STEMS_MIN_POINTS: " + str(find_stems_min_points))


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
        "-m", str(find_stems_min_points)
    ])

    # run the command with the arguments
    print("Running the command with the arguments")
    RunCommand(cmd, args)()

    # compute the metric
    print("Computing the metric")
    metric = InstanceSegmentationMetricsInFolder(
        gt_las_folder_path = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/input_data',
        target_las_folder_path = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds_with_ground',
        verbose=True
    ) 

    f1_score = metric.main()
    print("F1 score: " + str(f1_score))

    # log the metric
    print("Logging the metric")
    wandb.log({"f1_score": f1_score})


# define the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep-train", entity="maciej-wielgosz-nibio")

# run the sweep
wandb.agent(sweep_id, function=main, count=1000)
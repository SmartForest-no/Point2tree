from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


from pipeline_test_command_params_just_tls import RunCommand
from pipeline_test_command_params_just_tls import main as pipeline_main

def bayes_opt_main(
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
    n_tiles = int(n_tiles)
    find_stems_min_points = int(find_stems_min_points)
    return pipeline_main(
        n_tiles,
        slice_thickness,
        find_stems_height,
        find_stems_thickness,
        graph_maximum_cumulative_gap,
        add_leaves_voxel_length,
        find_stems_min_points,
        graph_edge_length,
        add_leaves_edge_length
    )

pbounds = {
    'n_tiles': (3, 3),
    'slice_thickness': (0.25, 0.75),
    'find_stems_height': (0.5, 2.0),
    'find_stems_thickness': (0.1, 1.0),
    'graph_maximum_cumulative_gap': (5, 20),
    'add_leaves_voxel_length': (0.1, 0.5),
    'find_stems_min_points': (50, 500),
    'graph_edge_length': (0.1, 2.0),
    'add_leaves_edge_length': (0.2, 1.5)
}

optimizer = BayesianOptimization(
    f=bayes_opt_main,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True
)

# load the logs
load_logs(optimizer, logs=["./bayes_opt_run_logs.json"])

logger = JSONLogger(path="./bayes_opt_run_logs_1.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=1,
    n_iter=100
)

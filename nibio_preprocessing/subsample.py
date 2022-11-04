
import logging
import joblib
import numpy as np


def subsample(self, tensor_x_y_z, voxel_size=0.1):
    logging.info('Subsampling...')
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = \
        np.unique(((tensor_x_y_z - np.min(tensor_x_y_z, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted=np.argsort(inverse)
    voxel_grid={}
    grid_barycenter,grid_candidate_center=[],[]

    def grid_subsampling(non_empty_voxel_keys):
        last_seen=0
        for idx,vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)]=tensor_x_y_z[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
            grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
            grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]-np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
            last_seen+=nb_pts_per_voxel[idx]

        return grid_candidate_center

    # use joblib to parallelize the computation of the for loop
    grid_candidate_center = joblib.Parallel(n_jobs=12)(joblib.delayed(grid_subsampling)(non_empty_voxel_keys) for i in range(12))

    # merge the results
    grid_candidate_center = np.concatenate(grid_candidate_center, axis=0)

    grid_candidate_center = np.array(grid_candidate_center)
    new_points = grid_candidate_center.transpose()

    return new_points
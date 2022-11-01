
import glob
import argparse
import os
import joblib
import laspy
import logging

import numpy as np


logging.basicConfig(level=logging.INFO)

class InstanceSegmentationMetrics():
    def __init__(self, gt_folder, pred_folder):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder

    def get_metrics_for_single_point_cloud(self, las_gt, las_pred):
        # read las files
        las_gt = laspy.read(las_gt)
        las_pred = laspy.read(las_pred)

        # get different classes from gt and pred
        gt_classes = np.unique(las_gt.treeID)
        pred_classes = np.unique(las_pred.instance_nr)

        # put x, y, z, for different classes in a dictionary
        gt_dict = {}
        pred_dict = {}
        for gt_class in gt_classes:
            gt_dict[gt_class] = np.vstack((las_gt.x[las_gt.treeID == gt_class], las_gt.y[las_gt.treeID == gt_class], las_gt.z[las_gt.treeID == gt_class])).T
        for pred_class in pred_classes:
            pred_dict[pred_class] = np.vstack((las_pred.x[las_pred.instance_nr == pred_class], las_pred.y[las_pred.instance_nr == pred_class], las_pred.z[las_pred.instance_nr == pred_class])).T

        # get the number of points in gt and pred per class and put it in a dictionary
        gt_dict_points = {}
        pred_dict_points = {}
        for gt_class in gt_classes:
            gt_dict_points[gt_class] = gt_dict[gt_class].shape[0]
        for pred_class in pred_classes:
            pred_dict_points[pred_class] = pred_dict[pred_class].shape[0]
            
        # compute overlap for each class
        overlap = {}
        for gt_class in gt_dict:
            for pred_class in pred_dict:
                overlap[(gt_class, pred_class)] = self.get_overlap(gt_dict[gt_class], pred_dict[pred_class])

        # get number of overlapping points per class
        overlap_points = {}
        for gt_class in gt_dict:
            for pred_class in pred_dict:
                overlap_points[(gt_class, pred_class)] = np.sum(overlap[(gt_class, pred_class)])

        # sort the overlap points in descending order
        overlap_points = {k: v for k, v in sorted(overlap_points.items(), key=lambda item: item[1], reverse=True)}

        # sort out overlaps by the number of points in gt and pred
        sorted_overlap = sorted(overlap.items(), key=lambda x: x[1], reverse=True)
        sorted_overlap_points = sorted(overlap_points.items(), key=lambda x: x[1], reverse=True)

     
        # # find overlap between gt 0 and 0 pred classes using subsampling and voxel size 0.5
        # logging.info('Overlap between gt 0 and pred 0 using subsampling and voxel size 0.5: {}'.format(self.get_overlap(gt_dict[0], pred_dict[0], subsample=True, voxel_size=0.5)))

        # # find overlap between gt 0 and 0 pred classes using subsampling and voxel size 0.05
        # logging.info('Overlap between gt 0 and pred 0 using subsampling and voxel size 0.05: {}'.format(self.get_overlap(gt_dict[0], pred_dict[0], subsample=True, voxel_size=0.05)))

        # # find overlap between gt 0 and 0 pred classes without subsampling
        # logging.info('Overlap between gt 0 and pred 0 without subsampling: {}'.format(self.get_overlap(gt_dict[0], pred_dict[0], subsample=False)))

        # find overlap between gt 0 and and other pred classes using subsampling and voxel size 0.5
        for i in range(1, len(pred_dict)):
            logging.info('Overlap between gt 0 and pred {} using subsampling and voxel size 0.5: {}'.format(i, self.get_overlap(gt_dict[0], pred_dict[i], subsample=True, voxel_size=0.5)))

        # find overlap between gt 9 and and other pred classes using subsampling and voxel size 0.5
        for i in range(1, len(pred_dict)):
            logging.info('Overlap between gt 9 and pred {} using subsampling and voxel size 0.5: {}'.format(i, self.get_overlap(gt_dict[9], pred_dict[i], subsample=True, voxel_size=0.1)))
       

        # print sorted overlap for first 10 classes
        # logging.info('Sorted overlap for classes: {}'.format(sorted_overlap))

        # # print best match for classes along with overlap
        # logging.info('Best match for classes: {}'.format(best_match))

    def get_overlap(self, gt, pred, subsample=False, voxel_size=0.1):
        # compute overlap between gt and pred

        # if subsample is True, subsample the point cloud
        if subsample:
            gt = self.subsample(gt, voxel_size)
            pred = self.subsample(pred, voxel_size)

        # get the number of points in gt and pred
        overlap = np.intersect1d(gt, pred).shape[0]
        # overlap = np.intersect1d(gt, pred).shape[0]

        # overlap = np.sum(np.all(gt[:, None] == pred, axis=-1), axis=0)
        return overlap

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
        

    def get_metrics_for_all_point_clouds(self):
        # get all las files in gt and pred folders using glob
        las_gt = glob.glob(os.path.join(self.gt_folder, '*.las'))
        las_pred = glob.glob(os.path.join(self.pred_folder, '*.las'))

        # if the number of files in gt and pred are not the same, raise an exception
        if len(las_gt) != len(las_pred):
            raise Exception('Number of files in gt and pred folders are not the same.')
        
        # iterate over all files in gt and pred folders
        for i in range(len(las_gt)):
            self.get_metrics_for_single_point_cloud(las_gt[i], las_pred[i])

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gt_folder', type=str, required=True, help='Path to the folder containing ground truth point clouds.')
    # parser.add_argument('--pred_folder', type=str, required=True, help='Path to the folder containing predicted point clouds.')
    # args = parser.parse_args()

    # # create an instance of InstanceSegmentationMetrics class
    # instance_segmentation_metrics = InstanceSegmentationMetrics(args.gt_folder, args.pred_folder)

    input_file_path = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/input_data'
    instance_segmented_file_path = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/sample_playground/results/instance_segmented_point_clouds_with_ground'

    instance_segmentation_metrics = InstanceSegmentationMetrics(input_file_path, instance_segmented_file_path)

    instance_segmentation_metrics.get_metrics_for_all_point_clouds()

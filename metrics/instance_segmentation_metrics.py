
import glob
import argparse
import os
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

        # compute overlap for each class
        overlap = {}
        for gt_class in gt_dict:
            for pred_class in pred_dict:
                overlap[(gt_class, pred_class)] = self.get_overlap(gt_dict[gt_class], pred_dict[pred_class])

        # print the number of classes in gt and pred
        logging.info('Number of classes in gt: {}'.format(len(gt_classes)))
        logging.info('Number of classes in pred: {}'.format(len(pred_classes)))

        # print first 10 classes in gt and pred
        logging.info('First 10 classes in gt: {}'.format(gt_classes[:10]))
        logging.info('First 10 classes in pred: {}'.format(pred_classes[:10]))

        # print overlap for first 10 classes
        logging.info('Overlap for first 10 classes: {}'.format(overlap))


    def get_overlap(self, gt, pred):
        # compute overlap between gt and pred
        overlap = np.intersect1d(gt, pred).shape[0]
        # overlap = np.sum(np.all(gt[:, None] == pred, axis=-1), axis=0)
        return overlap

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to the folder containing ground truth point clouds.')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the folder containing predicted point clouds.')
    args = parser.parse_args()

    # create an instance of InstanceSegmentationMetrics class
    instance_segmentation_metrics = InstanceSegmentationMetrics(args.gt_folder, args.pred_folder)
    instance_segmentation_metrics.get_metrics_for_all_point_clouds()

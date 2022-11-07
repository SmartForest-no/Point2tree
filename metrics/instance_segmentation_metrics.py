

import argparse
import os
import laspy
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

logging.basicConfig(level=logging.INFO)

class InstanceSegmentationMetrics:
    def __init__(
        self, 
        input_file_path, 
        instance_segmented_file_path, 
        save_to_csv=False,
        verbose=False
        ) -> None:

        self.input_file_path = input_file_path
        self.instance_segmented_file_path = instance_segmented_file_path
        self.save_to_csv = save_to_csv
        self.verbose = verbose
        # read and prepare input las file and instance segmented las file
        self.input_las = laspy.read(self.input_file_path)
        self.instance_segmented_las = laspy.read(self.instance_segmented_file_path)
        # get labels from input las file
        self.X_labels = self.input_las.treeID.astype(int) #TODO: generalize this to other labels
        # get labels from instance segmented las file
        self.Y_labels = self.instance_segmented_las.instance_nr.astype(int) #TODO: generalize this to other labels
        self.dict_Y = self.do_knn_mapping()

    def do_knn_mapping(self):
        X = self.input_las.xyz
        Y = self.instance_segmented_las.xyz
        X_labels = self.X_labels
        Y_labels = self.Y_labels

        # create a KDTree for X
        tree = KDTree(X, leaf_size=50, metric='euclidean')       
        # query the tree for Y     
        ind = tree.query(Y, k=1, return_distance=False)   

        # get labels for ind
        ind_labels_Y = X_labels[ind]
        # reshape to 1D
        ind_labels_Y = ind_labels_Y.reshape(-1) # labels from X matched to Y

        # create a dictionary which contains Y, Y_labels and ind_labels_Y
        dict_Y = {'Y': Y, 'Y_labels': Y_labels, 'ind_labels_Y': ind_labels_Y}

        return dict_Y

    def get_dominant_lables_sorted(self):
        # get unique labels from Y_labels
        Y_unique_labels = np.unique(self.Y_labels)
    
        dominant_labels = {}
        for label in Y_unique_labels:
            # get the indices of Y_labels == label
            ind_Y_labels = np.where(self.Y_labels == label)[0]
            # get the ind_labels_Y for these indices
            ind_labels_Y = self.dict_Y['ind_labels_Y'][ind_Y_labels]
            # get the unique ind_labels_Y
            unique_ind_labels_Y = np.unique(ind_labels_Y)
            # print the number of points for each unique ind_labels_Y
            tmp_dict = {}
            for unique_ind_label_Y in unique_ind_labels_Y:
                # get the indices of ind_labels_Y == unique_ind_label_Y
                ind_ind_labels_Y = np.where(ind_labels_Y == unique_ind_label_Y)[0]
                # put the number of points to the tmp_dict
                tmp_dict[str(unique_ind_label_Y)] = ind_ind_labels_Y.shape[0]
        
            # put the dominant label to the dominant_labels
            dominant_labels[str(label)] = tmp_dict

        # sort dominant_labels by the number of points
        dominant_labels_sorted = {}
        for key, value in dominant_labels.items():
            dominant_labels_sorted[key] = {k: v for k, v in sorted(value.items(), key=lambda item: item[1], reverse=True)}

        # iterate over the dominant_labels_sorted and sort it based on the first value of sub-dictionary
        dominant_labels_sorted = {
            k: v for k, v in sorted(dominant_labels_sorted.items(), key=lambda item: list(item[1].values())[0], reverse=True)}

        return dominant_labels_sorted

    def extract_from_sub_dict(self, target_dict, label):
        new_dict = {}

        for key_outer, value_outer in target_dict.items():
            tmp_dict = {}
            for item_inner in value_outer.keys():
                if item_inner == label:
                    tmp_dict[item_inner] = value_outer[item_inner]

            new_dict[key_outer] = (tmp_dict)
        return new_dict


    # define a function that finds class in input_file with the most points
    def find_dominant_classes_in_gt(self, input_file):
        # get the unique labels
        unique_labels = np.unique(input_file.treeID)
        # create a dictionary
        tmp_dict = {}
        for label in unique_labels:
            # get the indices of input_file.treeID == label
            ind_label = np.where(input_file.treeID == label)[0]
            # put the number of points to the tmp_dict
            tmp_dict[str(label)] = ind_label.shape[0]
        # sort tmp_dict by the number of points
        tmp_dict_sorted = {k: v for k, v in sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True)}
        return tmp_dict_sorted.keys()

    def get_the_dominant_label(self, dominant_labels_sorted):
        dominant_label_key = list(dominant_labels_sorted.keys())[0]
        dominant_label_value = list(dominant_labels_sorted.values())[0]
        dominant_label = list(dominant_label_value.keys())[0]

        # get dominant_label_key and dominant_label for which dominant_label_value.values() has the highest value
        for key, value in dominant_labels_sorted.items():
            for item in value.keys():
                if value[item] > dominant_label_value[dominant_label]:
                    dominant_label_key = key
                    dominant_label = item
                    dominant_label_value = value

        return dominant_label_key, dominant_label


    def remove_dominant_label(self, dominant_labels_sorted, dominant_label_key, dominant_label):
        # remove the dominant_label_key from dominant_labels_sorted
        dominant_labels_sorted.pop(dominant_label_key)
        # remove the dominant_label from the sub-dictionary of dominant_labels_sorted
        for key, value in dominant_labels_sorted.items():
            if dominant_label in value:
                value.pop(dominant_label)

        return dominant_labels_sorted


    def iterate_over_pc(self):

        label_mapping_dict = {}

        dominant_labels_sorted = self.get_dominant_lables_sorted()
        gt_classes_to_iterate = self. find_dominant_classes_in_gt(self.input_las)

        for gt_class in gt_classes_to_iterate:
            if len(dominant_labels_sorted) == 1:
                dominant_label_key, dominant_label = self.get_the_dominant_label(dominant_labels_sorted)
                label_mapping_dict[dominant_label_key] = dominant_label
                break

            dominant_label_key, dominant_label = self.get_the_dominant_label(
                self.extract_from_sub_dict(dominant_labels_sorted, gt_class))
        
            self.remove_dominant_label(dominant_labels_sorted, dominant_label_key, dominant_label)
            
            label_mapping_dict[dominant_label_key] = dominant_label
            
        # change keys and values to int
        label_mapping_dict = {int(k): int(v) for k, v in label_mapping_dict.items()}
        
        return label_mapping_dict

    def compute_metrics(self):
        # get the label_mapping_dict
        label_mapping_dict = self.iterate_over_pc()
        Y_unique_labels = np.unique(self.Y_labels)
        # map the labels
        metric_dict = {}

        for label in Y_unique_labels:
            # get the indices of Y_labels == label
            ind_Y_labels_label = np.where(self.Y_labels == label)[0]

            # get the ind_labels_Y for these indices
            ind_labels_Y = self.dict_Y['ind_labels_Y'][ind_Y_labels_label]

            # get the dominant label for this label
            dominant_label = label_mapping_dict[label]

            # get the indices of ind_labels_Y == dominant_label
            ind_dominant_label = np.where(ind_labels_Y == dominant_label)[0]

            # true positive is the number of points for dominant_label
            true_positive = ind_dominant_label.shape[0]

            # false positive is the number of all the points of this dominant_label label minus the true positive
            false_positive = np.where(self.dict_Y['ind_labels_Y'] == dominant_label)[0].shape[0] - true_positive

            # false negative is the number of all the points in Y_labels minus the number of points of true_positive
            false_negative = np.where(ind_labels_Y != dominant_label)[0].shape[0] 

            # true negative is the number of all the points minus the number of points of true_positive and false_positive
            true_negative = self.dict_Y['ind_labels_Y'].shape[0] - false_negative - true_positive - false_positive

            # sum all the true_positive, false_positive, false_negative, true_negative
            sum_all = true_positive + false_positive + false_negative + true_negative

            # get precision
            precision = true_positive / (true_positive + false_positive)
            # get recall
            recall = true_positive / (true_positive + false_negative)
            # get f1 score
            f1_score = 2 * (precision * recall) / (precision + recall)
            # get IoU
            IoU = true_positive / (true_positive + false_positive + false_negative)

            # find hight of the tree in the ground truth
            hight_of_tree = (self.input_las[self.input_las.treeID == dominant_label].z).max() - (self.input_las[self.input_las.treeID == dominant_label].z).min()

            # create tmp dict
            tmp_dict = {
            'dominant_label': dominant_label,
            'high_of_tree': hight_of_tree,
            'sum_all': sum_all,
            'true_positive': true_positive, 
            'false_positive': false_positive, 
            'false_negative': false_negative, 
            'true_negative': true_negative,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'IoU': IoU,
            }
            metric_dict[str(label)] = tmp_dict

        # compute a singel f1 score weighted by the tree hight
        f1_score_weighted = 0
        for key, value in metric_dict.items():
            f1_score_weighted += value['f1_score'] * value['high_of_tree']

        f1_score_weighted = f1_score_weighted / self.input_las.z.max()
        f1_score_weighted = f1_score_weighted / len(Y_unique_labels)


        return metric_dict, f1_score_weighted

    def print_metrics(self, metric_dict):
        for key, value in metric_dict.items():
            print(f'Label: {key}')
            for key2, value2 in value.items():
                print(f'{key2}: {value2}')
            print('')

    def save_to_csv_file(self, metric_dict):
        df = pd.DataFrame(metric_dict).T
        df.to_csv('metrics_instance_segmentation.csv')


    def main(self):
        metric_dict, f1_score_weighted  = self.compute_metrics()

        if self.verbose:
            print(f'f1_score_weighted: {f1_score_weighted}')
            for key, value in metric_dict.items():
                print(f'Label: {key}, f1_score: {value["f1_score"]}, high_of_tree: {value["high_of_tree"]}')
       
        if self.save_to_csv:
            self.save_to_csv_file(metric_dict)
            print(f'CSV file saved to: {os.path.join(os.getcwd(), "metrics_instance_segmentation.csv")}')

        return metric_dict, f1_score_weighted


# main
if __name__ == '__main__':
    # do argparse input_file_path, instance_segmented_file_path, verbose
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', type=str, required=True)
    parser.add_argument('--instance_segmented_file_path', type=str, required=True)
    parser.add_argument('--save_to_csv', action='store_true', help="Save the metrics to a csv file", default=False)
    parser.add_argument('--verbose', action='store_true', help="Print information about the process", default=False)

    args = parser.parse_args()

    # create instance of the class InstanceSegmentationMetrics
    instance_segmentation_metrics = InstanceSegmentationMetrics(
        args.input_file_path, 
        args.instance_segmented_file_path, 
        args.save_to_csv,
        args.verbose
        )
    
    # compute metrics
    metric_dict = instance_segmentation_metrics.main()
   
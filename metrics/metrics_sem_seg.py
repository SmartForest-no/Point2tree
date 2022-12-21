import argparse
import os
from joblib import Parallel, delayed
import laspy
from matplotlib import pyplot as plt
import numpy as np
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score


class MetricSemSeg:
    """
    This function returns the metrics for semantic segmentation. It takes in the ground truth folder and the predicted folder.
    It returns the following metrics:
    1. f1 score
    2. precision score
    3. recall score
    4. confusion matrix

    The input point clouds should be in the las format.
    """
    def __init__(
        self, 
        gt_folder,
        pred_folder,
        adjust_gt_data=False,
        plot_confusion_matrix=False, 
        verbose=False
        ):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.adjust_gt_data = adjust_gt_data # if True, the ground truth data will be adjusted to match the predicted data
        # this means removing data from 0 class which are considered as not important and should not be considered in the metrics
        self.plot_confusion_matrix = plot_confusion_matrix
        self.verbose = verbose

    def get_file_name_list(self):

        # get list of the original point clouds
        file_name_list_original = []
        for file in os.listdir(self.gt_folder):
            if file.endswith(".las"):
                file_name_list_original.append(os.path.join(self.gt_folder, file))

        # sort the list in place
        file_name_list_original.sort()

        # get the list of the core names of the original point clouds
        # file_name_list_original_core = []
        # for file_name in file_name_list_original:
        #     file_name_list_original_core.append(os.path.basename(file_name).split('.')[0])

        # get list of the predicted point clouds
        file_name_list_predicted = []
        for file in os.listdir(self.pred_folder):
            if file.endswith(".las"):
                file_name_list_predicted.append(os.path.join(self.pred_folder, file))

        # sort the list in place
        file_name_list_predicted.sort()

        # check if the number of files in the two folders is the same
        if len(file_name_list_original) != len(file_name_list_predicted):
            raise Exception('The number of files in the two folders is not the same.')

        # get core names of ground truth point clouds
        file_name_list_original_core = []
        for file_name in file_name_list_original:
            file_name_list_original_core.append(os.path.basename(file_name).split('.')[0])

        # get core names of predicted point clouds
        file_name_list_predicted_core = []
        for file_name in file_name_list_predicted:
            file_name_list_predicted_core.append(os.path.basename(file_name).split('.')[0])

        # remove the suffix of '.segmented' from the predicted point clouds
        file_name_list_predicted_core = [file_name.split('.')[0] for file_name in file_name_list_predicted_core]

        # compare the two lists and check if they are the same
        if file_name_list_original_core != file_name_list_predicted_core:
            raise ValueError("The two lists of point clouds are not the same")


        # zip the two lists
        file_name_list = list(zip(file_name_list_original, file_name_list_predicted))

        return file_name_list

    def get_labels_from_point_file(self, file_name, predicted=False):
        """
        This function returns the labels of a point cloud file.
        """
        point_cloud = laspy.read(file_name)
        if self.adjust_gt_data and not predicted:
            if self.verbose:
                print("adjusting ground truth data")
            labels = point_cloud.label
            labels = labels[labels != 0]
            labels = labels - 1
        else:
            labels = point_cloud.label

        # convert to int
        labels = labels.astype(int)

        return labels

    def get_xyz_from_point_file(self, file_name, predicted=False):
        """
        This function returns the xyz coordinates of a point cloud file.
        """
        point_cloud = laspy.read(file_name)
        if self.adjust_gt_data and not predicted:
            if self.verbose:
                print("adjusting ground truth data")
            # find all the points xyz with label 0
            labels = point_cloud.label
            xyz = np.vstack((
                point_cloud.x[labels !=0], 
                point_cloud.y[labels !=0], 
                point_cloud.z[labels !=0]
                )).transpose()

        else:
            xyz = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

        return xyz

    def get_metrics_for_single_file(self, file_name_original, file_name_predicted):
        """
        This function returns the confusion matrix for a single point cloud and also the precision, recall and f1 score.
        """

        # get labels
        labels_predicted = self.get_labels_from_point_file(file_name_predicted, predicted=True)
        labels_original = self.get_labels_from_point_file(file_name_original)

        # find and print the ranges of the labels
        if self.verbose:
            print("labels_predicted range: ", np.min(labels_predicted), np.max(labels_predicted))
            print("labels_original range: ", np.min(labels_original), np.max(labels_original))

        # print shape of labels
        if self.verbose:
            print("labels_predicted.shape: ", labels_predicted.shape)
            print("labels_original.shape: ", labels_original.shape)


        # get points
        xyz_predicted = self.get_xyz_from_point_file(file_name_predicted, predicted=True)
        xyz_original = self.get_xyz_from_point_file(file_name_original)


        # find the closest point in the original point cloud for each point in the predicted point cloud using the euclidean distance using knn

        # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xyz_original)
        # distances, indices = nbrs.kneighbors(xyz_predicted)

        tree = KDTree(xyz_original, leaf_size=50, metric='euclidean')       
        # query the tree for Y     
        indices = tree.query(xyz_predicted, k=1, return_distance=False)   

        # get the labels of the closest points
        labels_original_closest = labels_original[indices]

        # get the confusion matrix
        conf_matrix = np.round(confusion_matrix(labels_original_closest, labels_predicted, normalize='false'), decimals=2)

        # if conf_matrix.shape[0] == 3 add diagonal elements to make it 4x4 at dimension 2
        if conf_matrix.shape[0] == 3:
            if self.verbose:
                print("conf_matrix.shape[0] == 3 expanding to 4x4 at dimension 2")
            conf_matrix = np.insert(conf_matrix, 2, 0, axis=1)
            conf_matrix = np.insert(conf_matrix, 2, 0, axis=0)

        # print the confusion matrix shape
        if self.verbose:
            print("conf_matrix.shape: ", conf_matrix.shape)

        # get picture of the confusion matrix
        if self.plot_confusion_matrix:
            if conf_matrix.shape[0] == 3:
                class_names = ['terrain', 'vegetation', 'stem']
            elif conf_matrix.shape[0] == 4:
                class_names = ['terrain', 'vegetation', 'CWD', 'stem']
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
            disp.plot()
            plt.savefig('confusion_matrix.png')

        # compute precision, recall and f1 score using sklearn.metrics 
        precision = precision_score(labels_original_closest, labels_predicted, average='weighted')
        precision = np.round(precision, decimals=3)
        recall = recall_score(labels_original_closest, labels_predicted, average='weighted')
        recall = np.round(recall, decimals=3)
        f1 = f1_score(labels_original_closest, labels_predicted, average='weighted')
        f1 = np.round(f1, decimals=3)

        # compute precision, recall and f1 per class
        precision_per_class = precision_score(labels_original_closest, labels_predicted, average=None)  
        precision_per_class = np.round(precision_per_class, decimals=3)
        recall_per_class = recall_score(labels_original_closest, labels_predicted, average=None)
        recall_per_class = np.round(recall_per_class, decimals=3)
        f1_per_class = f1_score(labels_original_closest, labels_predicted, average=None)
        f1_per_class = np.round(f1_per_class, decimals=3)

        # put all the results in a dictionary
        results = {
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }

        return results

    def get_metrics_for_all_files(self):
        """
        This function returns the confusion matrix for all point clouds and also the precision, recall and f1 score.
        """
        file_name_list = self.get_file_name_list()

        # create empty lists for the results
        conf_matrix_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        precision_per_class_list = []
        recall_per_class_list = []
        f1_per_class_list = []

        # loop over all files
        # results = []
        # for file_name_original, file_name_predicted in tqdm(file_name_list):
        #     if self.verbose:
        #         print("file_name_original: ", file_name_original)
        #         print("file_name_predicted: ", file_name_predicted)

        #     results.append(self.get_metrics_for_single_file(file_name_original, file_name_predicted))

        # parallelize the computation
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(self.get_metrics_for_single_file)(file_name_original, file_name_predicted) for file_name_original, file_name_predicted in file_name_list
        )

        # extract the results from the dictionary
        for results in results:
            conf_matrix_list.append(results['confusion_matrix'])
            precision_list.append(results['precision'])
            recall_list.append(results['recall'])
            f1_list.append(results['f1'])
            precision_per_class_list.append(results['precision_per_class'])
            recall_per_class_list.append(results['recall_per_class'])
            f1_per_class_list.append(results['f1_per_class'])

        # compute the mean of the confusion matrix
        conf_matrix_mean = np.mean(conf_matrix_list, axis=0)

        # compute the mean of the precision, recall and f1 score
        precision_mean = np.mean(precision_list)
        recall_mean = np.mean(recall_list)
        f1_mean = np.mean(f1_list)

        # compute the mean of the precision, recall and f1 score per class
        # compute separately for items which have 3 and 4 classes
        # create a separate list for item with 3 classes
        precision_per_class_list_3 = []
        recall_per_class_list_3 = []
        f1_per_class_list_3 = []
        # create a separate list for item with 4 classes
        precision_per_class_list_4 = []
        recall_per_class_list_4 = []
        f1_per_class_list_4 = []
        # loop over all items
        for i in range(len(precision_per_class_list)):
            # check if the item has 3 classes
            if len(precision_per_class_list[i]) == 3:
                precision_per_class_list_3.append(precision_per_class_list[i])
                recall_per_class_list_3.append(recall_per_class_list[i])
                f1_per_class_list_3.append(f1_per_class_list[i])
            # check if the item has 4 classes
            elif len(precision_per_class_list[i]) == 4:
                precision_per_class_list_4.append(precision_per_class_list[i])
                recall_per_class_list_4.append(recall_per_class_list[i])
                f1_per_class_list_4.append(f1_per_class_list[i])
            else:
                print("ERROR: the number of classes is neither 3 nor 4")
                exit()


        # compute the mean of the precision, recall and f1 score per class for items with 3 classes
        precision_per_class_mean_3 = np.mean(precision_per_class_list_3, axis=0)
        recall_per_class_mean_3 = np.mean(recall_per_class_list_3, axis=0)
        f1_per_class_mean_3 = np.mean(f1_per_class_list_3, axis=0)

        # compute the mean of the precision, recall and f1 score per class for items with 4 classes
        precision_per_class_mean_4 = np.mean(precision_per_class_list_4, axis=0)
        recall_per_class_mean_4 = np.mean(recall_per_class_list_4, axis=0)
        f1_per_class_mean_4 = np.mean(f1_per_class_list_4, axis=0)

        # compute the mean of the precision, recall and f1 score per class
        precision_per_class_mean = np.mean(np.concatenate((precision_per_class_mean_3, precision_per_class_mean_4), axis=0), axis=0)
        recall_per_class_mean = np.mean(np.concatenate((recall_per_class_mean_3, recall_per_class_mean_4), axis=0), axis=0)
        f1_per_class_mean = np.mean(np.concatenate((f1_per_class_mean_3, f1_per_class_mean_4), axis=0), axis=0)

        # put all the results in a dictionary
        results = {
            'confusion_matrix': conf_matrix_mean,
            'precision': precision_mean,
            'recall': recall_mean,
            'f1': f1_mean,
            'precision_per_class': precision_per_class_mean,
            'recall_per_class': recall_per_class_mean,
            'f1_per_class': f1_per_class_mean
        }

        return results
      
    def main(self):
        # get the metrics for all point clouds
        if self.verbose:
            print("get the metrics for all point clouds")

        results = self.get_metrics_for_all_files()

        if self.verbose:
            print("results: ", results)

        return results

if __name__ == '__main__':
    # use argparse to parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_original', type=str, default='data/original', help='path to the original point clouds directory')
    parser.add_argument('--path_predicted', type=str, default='data/predicted', help='path to the predicted point clouds directory')
    parser.add_argument('--adjust_gt_data', action='store_true', default=False,  help='adjust the ground truth data')
    parser.add_argument('--plot_confusion_matrix', action='store_true', default=False,  help='plot the confusion matrix')
    parser.add_argument('--verbose', action='store_true', default=False,  help='print more information')
    args = parser.parse_args()

    # create an instance of the class
    metrics = MetricSemSeg(
        args.path_original, 
        args.path_predicted, 
        args.adjust_gt_data,
        args.plot_confusion_matrix, 
        args.verbose
        )

    # get the metrics
    results = metrics.main()








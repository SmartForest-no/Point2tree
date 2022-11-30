
import argparse
import os
import laspy
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score


class MetricSemSeg:
    def __init__(
        self, 
        folder_with_results, 
        plot_confusion_matrix=False, 
        suffix_of_the_folder_with_results='_FSCT_output', 
        verbose=False
        ):

        self.folder_with_results = folder_with_results
        self.plot_confusion_matrix = plot_confusion_matrix
        self.suffix_of_the_folder_with_results = suffix_of_the_folder_with_results
        self.verbose = verbose

    def get_file_name_list(self):
        """
        It is assumed that the original point clouds are in the folder with results.
        This function returns a list of file names of the original point clouds.
        """
        print("folder_with_results: ", self.folder_with_results)

        file_original_name_list = [f for f in os.listdir(self.folder_with_results) if os.path.isfile(os.path.join(self.folder_with_results, f))]
        file_original_name_list = [f for f in file_original_name_list if f.endswith('.las') or f.endswith('.laz')]

        # get prefix of the file name
        file_name_prefix_list = [f[:-4] for f in file_original_name_list]
        
        # create paths to the folders with results
        folder_with_results_list = [os.path.join(self.folder_with_results, f + self.suffix_of_the_folder_with_results) for f in file_name_prefix_list]
        # get files of the name "segmented_clean.las" from the folders with results
        file_predicted_name_list = [os.path.join(f, 'segmented_cleaned.las') for f in folder_with_results_list]
        # file_predicted_name_list = [os.path.join(f, 'segmented.las') for f in folder_with_results_list]

        # get full paths to the original files
        file_original_name_list = [os.path.join(self.folder_with_results, f) for f in file_original_name_list]

        # put original and predicted files into a tuple
        file_name_list = []

        for i in range(len(file_original_name_list)):
            file_name_list.append((file_original_name_list[i], file_predicted_name_list[i]))
   
        return file_name_list

    def get_labels_from_point_file(self, file_name):
        """
        This function returns the labels of a point cloud file.
        """
        point_cloud = laspy.read(file_name)
        labels = point_cloud.label
        return labels

    def get_xyz_from_point_file(self, file_name):
        """
        This function returns the xyz coordinates of a point cloud file.
        """
        point_cloud = laspy.read(file_name)
        xyz = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
        return xyz

    def get_metrics_for_single_file(self, file_name_original, file_name_predicted):
        """
        This function returns the confusion matrix for a single point cloud and also the precision, recall and f1 score.
        """

        # get labels
        labels_original = self.get_labels_from_point_file(file_name_original)
        labels_predicted = self.get_labels_from_point_file(file_name_predicted)

        # get points
        xyz_original = self.get_xyz_from_point_file(file_name_original)
        xyz_predicted = self.get_xyz_from_point_file(file_name_predicted)

        # find the closest point in the original point cloud for each point in the predicted point cloud using the euclidean distance using knn
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xyz_original)
        distances, indices = nbrs.kneighbors(xyz_predicted)

        # get the labels of the closest points
        labels_original_closest = labels_original[indices]

        # get the confusion matrix
        conf_matrix = np.round(confusion_matrix(labels_original_closest, labels_predicted, normalize='true'), decimals=2)

        # get picture of the confusion matrix
        if self.plot_confusion_matrix:
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
        for file_name_original, file_name_predicted in tqdm(file_name_list):
            if self.verbose:
                print("file_name_original: ", file_name_original)
                print("file_name_predicted: ", file_name_predicted)
                
            results = self.get_metrics_for_single_file(file_name_original, file_name_predicted)
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
        precision_per_class_mean = np.mean(precision_per_class_list, axis=0)
        recall_per_class_mean = np.mean(recall_per_class_list, axis=0)
        f1_per_class_mean = np.mean(f1_per_class_list, axis=0)

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
    parser.add_argument('--folder_with_results', type=str, default='original', help='directory of the original point clouds')
    parser.add_argument('--plot_confusion_matrix', action='store_true', default=False, help='plot the confusion matrix')
    parser.add_argument('--suffix_of_the_folder_with_results', type=str, default='_FSCT_output', help='suffix of the directory of the predicted point clouds')
    parser.add_argument('--verbose', action='store_true', default=False,  help='print more information')
    args = parser.parse_args()

    # create an instance of the class
    metrics = MetricSemSeg(
        args.folder_with_results, 
        args.plot_confusion_matrix, 
        args.suffix_of_the_folder_with_results,
        args.verbose
        )
    results = metrics.main()








# This is evalutation script for the model which assumes a certain folder structure
import argparse
import os
import laspy
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class Eval():
    def __init__(self, folder, plot_confusion_matrix=False):
        self.folder = folder
        self.plot_confusion_matrix = plot_confusion_matrix
        
    def get_file_name_list(self):
        """
        This function returns a list of file names of 
        the original point clouds and a list of folder names of the predicted point clouds.
        """
        file_name_list = [f for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f))]
        file_name_list = [f for f in file_name_list if f.endswith('.las') or f.endswith('.laz')]
        folder_list = [f[:-4] for f in file_name_list]
        folder_list = [f + '_FSCT_output' for f in folder_list]
        if len(folder_list) == 0:
            raise Exception('No point cloud files found in the folder')
        if len(folder_list) != len(file_name_list):
            raise Exception('The number of point cloud files and folders are not equal')
        return file_name_list, folder_list

    def get_labels_from_cloud_point_file(self, file_name):
        """
        This function returns the labels of a point cloud file.
        """
        original_point_cloud = laspy.read(os.path.join(self.folder, file_name))
        labels = original_point_cloud.label
        return labels

    def get_labels_of_orignal_point_clouds(self):
        """
        This function returns the labels of the original point clouds.
        """
        file_name_list = self.get_file_name_list()[0]
        labels = [self.get_labels_from_cloud_point_file(file_name) for file_name in file_name_list]
        return labels

    def get_labels_of_predicted_point_clouds(self):
        """
        This function returns the labels of the predicted point clouds.
        """

        folder_list = self.get_file_name_list()[1]
        labels = [self.get_labels_from_cloud_point_file(os.path.join(folder, 'segmented.las')) for folder in folder_list]
        return labels

    def get_confusion_matrix_for_all_point_clouds(self):
        """
        This function returns the confusion matrix for all point clouds and also the average precision, recall and f1 score.
        """
        original_labels = self.get_labels_of_orignal_point_clouds()
        original_labels = np.hstack(original_labels) - 1
        predicted_labels = self.get_labels_of_predicted_point_clouds()
        predicted_labels = np.hstack(predicted_labels)

        # get the confusion matrix
        conf_matrix = np.round(confusion_matrix(original_labels, predicted_labels, normalize='true'), decimals=2)

        # get picture of the confusion matrix
        if self.plot_confusion_matrix:
            class_names = ['terrain', 'vegetation', 'CWD', 'stem']
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
            disp.plot()
            plt.savefig('confusion_matrix.png')


        # compute precision, recall and f1 score using sklearn.metrics 
        precision = precision_score(original_labels, predicted_labels, average='weighted')
        recall = recall_score(original_labels, predicted_labels, average='weighted')
        f1 = f1_score(original_labels, predicted_labels, average='weighted')

        # compute precision, recall and f1 per class
        precision_per_class = precision_score(original_labels, predicted_labels, average=None)
        recall_per_class = recall_score(original_labels, predicted_labels, average=None)
        f1_per_class = f1_score(original_labels, predicted_labels, average=None)

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
   
def main(folder, plot_confusion_matrix=False):
    eval = Eval(folder, plot_confusion_matrix=plot_confusion_matrix)
    metrics = eval.get_confusion_matrix_for_all_point_clouds() 
    return metrics

if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data", help="path to folder")
    parser.add_argument("--plot_confusion_matrix", type=bool, default=False, help="plot confusion matrix")
    args = parser.parse_args()

    # get the confusion matrix
    metrics = main(args.dir, plot_confusion_matrix=args.plot_confusion_matrix)   
    # print pretty the metrics
    print('Confusion matrix: ')
    print(metrics['confusion_matrix'])
    print('')
    print('Precision: ', metrics['precision'])
    print('Recall: ', metrics['recall'])
    print('F1: ', metrics['f1'])
    print('')
    print('Precision per class: ', metrics['precision_per_class'])
    print('Recall per class: ', metrics['recall_per_class'])
    print('F1 per class: ', metrics['f1_per_class'])

 
# partially based on : https://medium.com/analytics-vidhya/feature-importance-explained-bfc8d874bcf
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class FindParamImportance:
    def __init__(
        self, 
        logs_json_file, 
        plot_file_path='feature_importance.png',
         verbose=False
         ) -> None:
        self.logs_json_file = logs_json_file
        self.plot_file_path = plot_file_path
        self.verbose = verbose

    def get_data(self):
        runs = []
        for line in open(self.logs_json_file, 'r'):
            runs.append(json.loads(line))

        # get header of the logs
        header = ['target']

        for key in runs[0]['params']:
            header.append(key)

        # create a dictionary from the header with empty lists
        data = {}
        data = data.fromkeys(header)
        data = {key: [] for key in header}

        # fill the dictionary with the data
        for run in runs:
            data['target'].append(run['target'])
            for key in run['params']:
                data[key].append(run['params'][key])

        # create a dataframe from the dictionary
        df = pd.DataFrame(data)

        return df

    def get_feature_importance(self, df):
        # get the features
        X = df.drop('target', axis=1)
        # get the target
        y = df['target']

        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)

        model=LinearRegression()
        model.fit(X_scaled,y)
        importance=model.coef_

        # combine importance with the feature names
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(importance)})        
        # sort the values
        feature_importance.sort_values(by='importance', ascending=True, inplace=True)

        if self.verbose:
            print(feature_importance)

        return feature_importance


    def cluster_results_kmeans(self):
        # use k-means to cluster the results parameters into 4 groups
        K = 7

        data = self.get_data()
     
        X = data.drop('target', axis=1)
        # get the target
        y = data['target']

        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)

        kmeans = KMeans(n_clusters=K, random_state=0).fit(X_scaled)
        labels = kmeans.labels_

        # find mean values of y for each cluster
        y_means = []
        for i in range(K):
            y_means.append(y[labels == i].mean())

        # sort the clusters by the mean values of y
        print(y_means)


        print(labels)
        # plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=50, cmap='viridis')
        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        # add cluster labels to the plot
        for i in range(K):
            plt.text(X_scaled[labels == i, 0].mean(), X_scaled[labels == i, 1].mean(), str(i), ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
        plt.savefig('kmeans.png')

        # compute mean values of y for each cluster
        y_means = []
        for i in range(4):
            y_means.append(y[labels == i].mean())


        return labels

    def cluster_dbscan(self):
        data = self.get_data()
        X = data.drop('target', axis=1)
        # get the target
        y = data['target']

        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)

        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=2, min_samples=3).fit(X_scaled)
        labels = db.labels_

        print("db scan labels: ", labels)

        # find mean values of y for each cluster
        y_means = []
        # find number of unique labels
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        for i in range(n_clusters_):
            y_means.append(y[labels == i].mean())

        print("db scan: ", y_means)


        # plot the results the most important features
        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=50, cmap='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        # add cluster labels to the plot
        for i in range(n_clusters_):
            plt.text(X_scaled[labels == i, 0].mean(), X_scaled[labels == i, 1].mean(), str(i), ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
        plt.savefig('dbscan.png')

    def find_correlation(self):
        data = self.get_data()

        # get the features
        X = data.drop('target', axis=1)
        # get the target
        y = data['target']

        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)

        # remove from X_scaled the columns that have nan values
        X_scaled = X_scaled[:, ~np.isnan(X_scaled).any(axis=0)]
        # replace nan values with 0
        X_scaled = np.nan_to_num(X_scaled)

        # compute the correlation matrix and put values in the figur
        corr = np.corrcoef(X_scaled.T)
        # plot the correlation matrix
        plt.figure(figsize=(10, 6))
        plt.imshow(corr, cmap='viridis')
        plt.colorbar()
        plt.xticks(range(len(X.columns)), X.columns, rotation='vertical')
        plt.yticks(range(len(X.columns)), X.columns)
        # add correlation values to the plot
        for i in range(len(X_scaled.T)):
            for j in range(len(X_scaled.T)):
                plt.text(i, j, round(corr[i, j], 2), ha='center', va='center', color='white')
        plt.savefig('correlation.png')

        # find the most correlated features
        # get the upper triangle of the correlation matrix
        upper = np.triu(corr)
        # find the indices of the upper triangle that are not zero
        # these are the indices of the correlated features
        correlated_features = np.where(upper > 0.15)
        # get the feature names
        feature_names = X.columns
        # print the correlated features
        print('Correlated features:')
        for i in range(len(correlated_features[0])):
            if correlated_features[0][i] != correlated_features[1][i]:
                print(feature_names[correlated_features[0][i]], feature_names[correlated_features[1][i]])

    def find_correlation_with_the_output(self):
        data = self.get_data()
        # get the features
        X = data.drop('target', axis=1)
        # get the target
        y = data['target']

        # find how correlated each feature is with the target
        correlations = []
        for i in range(len(X.columns)):
            correlations.append(np.corrcoef(X.iloc[:, i], y)[0, 1])

        # plot the correlations and make text to fit in the plot
        plt.figure(figsize=(10, 6))
        plt.bar(X.columns, correlations)
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        # create a legend
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=0.15, color='red', linestyle='--')
        plt.axhline(y=-0.15, color='red', linestyle='--')
        plt.legend(['0', '0.15', '-0.15'])
        plt.title('Correlation with the output')
        plt.savefig('correlation_with_the_output.png')

    
     

    def find_highest_params_values(self):
        data = self.get_data()

        # get mean values of the features for 6 the highest values of the target
        # get the features
        X = data.drop('target', axis=1)
        # get the target
        y = data['target']

        # sort the target values
        y_sorted = y.sort_values(ascending=False)
        # get the indices of the 6 highest values
        indices = y_sorted.index[:4]
        # get the features for the 6 highest values
        X_highest = X.loc[indices]
        # get the mean values of the features
        X_highest_mean = X_highest.mean()

        # print the features with the highest mean values
        print(' ')
        print('Features with the highest mean values:')
        print('Feature name', 'Mean value')
        for i in range(len(X_highest_mean)):
            print(X_highest_mean.index[i], X_highest_mean[i])


    def gen_plot_of_feature_importance(self, feature_importance):
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        # reduce the font size of the labels
        plt.yticks(fontsize=6)
        plt.title('Feature Importance')

        # save the plot
        plt.savefig(self.plot_file_path)


    def main(self):
        df = self.get_data()
        feature_importance = self.get_feature_importance(df)
        self.gen_plot_of_feature_importance(feature_importance)
        if self.verbose:
            print('Done')
            print('Plot saved to: ', self.plot_file_path)

        # cluster_labels = self.cluster_results_kmeans()
        # cluster_labels_dbscan = self.cluster_dbscan()
        self.find_correlation()
        self.find_highest_params_values()
        self.find_correlation_with_the_output()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_json_file', type=str, default='logs.json')
    parser.add_argument('--plot_file_path', type=str, default='feature_importance.png')
    parser.add_argument('--verbose', help="Print more information.", action="store_true")


    args = parser.parse_args()
    logs_json_file = args.logs_json_file
    plot_file_path = args.plot_file_path
    verbose = args.verbose

    find_param_importance = FindParamImportance(
        logs_json_file=logs_json_file,
        plot_file_path=plot_file_path,
        verbose=verbose
    )

    find_param_importance.main()
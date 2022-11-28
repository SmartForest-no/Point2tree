# partially based on : https://medium.com/analytics-vidhya/feature-importance-explained-bfc8d874bcf
import json
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
        for line in open('logs.json', 'r'):
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

    def gen_plot_of_feature_importance(self, feature_importance):
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
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
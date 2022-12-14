import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class PlotProgressJson:
    def __init__(
        self, 
        logs_json_file, 
        plot_file_path='progress_plot.png',
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

    def gen_plot_of_progress(self):

        df = self.get_data()

        # get the features
        X = df.drop('target', axis=1)
        # get the target
        y = df['target']

        plt.ylim([0.6, 0.8])

      
        # plot the progress
        plt.plot(y)
        plt.xlabel('Iteration')
        plt.ylabel('Target')
        plt.title('Progress')

        # add the grid
        plt.grid(True)

        # add the trendline
        z = np.polyfit(range(len(y)), y, 1)
        p = np.poly1d(z)
        plt.plot(p(range(len(y))), "r--")
         # add the legend
        plt.legend(['Target', 'Trendline'])
        
        # plot the trensline without zeros
        y_non_zero = [i for i in y if i != 0]
        z = np.polyfit(range(len(y_non_zero)), y_non_zero, 1)
        p = np.poly1d(z)
        plt.plot(p(range(len(y_non_zero))), "g--")
        # add the legend
        plt.legend(['Target', 'Trendline', 'Trendline without zeros'])

        # add the moving average
        y_moving_average = y.rolling(window=10).mean()
        plt.plot(y_moving_average, "y--")
        # add the legend
        plt.legend(['Target', 'Trendline', 'Trendline without zeros', 'Moving average'])

        # get new y and change 0 with average of two neighbours 
        y_new = y.copy()
        for i in range(len(y_new)):
            if y_new[i] == 0:
                y_new[i] = (y_new[i-1] + y_new[i+1]) / 2
        # use new y_new to plot a moving average
        y_moving_average = y_new.rolling(window=10).mean()
        plt.plot(y_moving_average, "m--")
        # add the legend
        plt.legend(['Target', 'Trendline', 'Trendline without zeros', 'Moving average', 'Moving average without zeros'])



       
    
        

        # add gradient information to the plot
        plt.text(0.5, 0.5, f"Gradient: {z[0]:.4f}", fontsize=12, transform=plt.gcf().transFigure)

        plt.savefig(self.plot_file_path)

    def main(self):
        self.gen_plot_of_progress()

        if self.verbose:
            print(f"Plot saved to {self.plot_file_path}")
            print("Done.")
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_json_file', type=str, default='logs.json')
    parser.add_argument('--plot_file_path', type=str, default='plot_progress_json.png')
    parser.add_argument('--verbose', help="Print more information.", action="store_true")


    args = parser.parse_args()
    logs_json_file = args.logs_json_file
    plot_file_path = args.plot_file_path
    verbose = args.verbose

    PlotProgressJson(logs_json_file, plot_file_path, verbose).main()
   
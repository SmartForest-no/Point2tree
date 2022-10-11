
# Orinal repo
For the orignal repo, please take a look there: https://github.com/philwilkes/FSCT



# Installation steps of the pipeline
The installation involves conda.

The steps to take on Ubuntu 20.04 machine on a linux machine or in a docker:

```
UBUNTU_VER=20.04
CONDA_VER=latest
OS_TYPE=x86_64

mkdir conda_installation
cd conda_installation
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
sudo bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
sudo groupadd anaconda_admin
sudo chown -R :anaconda_admin /miniconda
sudo chmod -R 775 /miniconda 
sudo adduser nibio anaconda_admin
conda update conda
conda init
conda create --name pdal-env python=3.8.13
conda activate pdal-env
conda install -c conda-forge pdal python-pdal

```
You should reboot shell session at this point. 
Next, you should clone the repo with the following command: 

```
git clone git@github.com:maciekwielgosz/FSCT.git
```
if you didn't exchange ssh keys you may need to use the following command:
 `git clone https://github.com/maciekwielgosz/FSCT.git`

# Running the pipeline with the NIBIO code

Once you clone the repo you should export the path (you should be in FSCT folder).

For linux (Ubuntu): `export PYTHONPATH='.' ` and make sure that activate the conda `conda activate pdal-env`.

For Windows you can check: <https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages>


## Running a whole pipeline
In order to run a whole pipeline run the following command: `./run_all.sh folder`. 

It may take <ins> very long time </ins> depending on your machine and the number of files and if you change the parameter if points density (by default its 150) to some lower number. The lower the number the bigger pointclounds are to be precessed and more time it may take. Keep in mind that at some point (for too low of the number) the pipeline may break. 

Make sure that you put the data in `*.las` format to this folder. If your files are in a different format e.g. `*.laz` you can use `python nibio_preprocessing/convert_files_in_folder.py --input_folder input_folder_name --output_folder output_folder las ` to convert your file to `*.las` format. 

The pipeline is composed of serveral steps and input parametes in `run_all.sh input_folder_name` should be set before the run. The default parameters are as follows:
```
CLEAR_INPUT_FOLDER=1  # 1: clear input folder, 0: not clear input folder
CONDA_ENV="pdal-env-1" # conda environment for running the pipeline

# Tiling parameters
N_TILES=3
SLICE_THICKNESS=0.5
FIND_STEMS_HEIGHT=1.5
FIND_STEMS_THICKNESS=0.5
GRAPH_MAXIMUM_CUMULATIVE_GAP=3
ADD_LEAVES_VOXEL_LENGTH=0.5
FIND_STEMS_MIN_POINTS=50
```
The stages are :
* reduction of the point clound size to the point where it has density of 150 points / square meter
* mapping to `*.ply` format, all the reducted`*.las` files are mapped and the orignal files are removed (the converted to `*ply` are kept)
* semantic segmentation,
* instance segmentation,
* consolidation of the results (each instance is seperate so they have to be consolidated into a single cloud point),
* postprocessing which puts everthing to a single folder in `input_folder/results`. 

Folder `input_folder/results` contain three subfolders: 
```
.
+--input_data
+--instance_segmented_point_clouds
+--segmented_point_clouds
```

## Running with sample files
The repo comes with sample file. You can use them to test your setup. To run the folow do:
```
chmod +x run_sample.sh
./run_sample.sh
```

# Installation steps of the pipeline
The installation involves conda.

The steps to take on Ubuntu 20.04 machine on a linux machine or in a docker:

```
UBUNTU_VER=20.04
CONDA_VER=latest
OS_TYPE=x86_64

mkdir conda_installation
cd conda_installation
curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
sudo bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
sudo groupadd anaconda_admin
sudo chown -R :anaconda_admin /miniconda
sudo chmod -R 775 /miniconda 
sudo adduser nibio anaconda_admin 
```
reboot of the docker may be need at this point
```
/miniconda/bin/conda update conda
/miniconda/bin/conda init

```
You should reboot shell session at this point.
```

conda create --name pdal-env python=3.8.13
conda activate pdal-env
conda install -c conda-forge pdal python-pdal


```
You should reboot shell session at this point. 
Next, you should clone the repo with the following command: 

```
git clone https://github.com/SmartForest-no/Point2tree
```


You have to install requirements for the repo.
 ```
 pip install -r requirements.txt
 ```

# Running the pipeline with the NIBIO code

Once you clone the repo you should export the path (you should be in FSCT folder).

For linux (Ubuntu): `export PYTHONPATH='.' ` and make sure that activate the conda `conda activate pdal-env`.

For Windows you can check: <https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages>


## Running a whole pipeline

It may take <ins> very long time </ins> depending on your machine and the number of files and if you change the parameter if points density (by default its 150) to some lower number. The lower the number the bigger pointclounds are to be precessed and more time it may take. Keep in mind that at some point (for too low of the number) the pipeline may break. 

The default model which is available in the repo in `fsct\model\model.path` was trained on the nibio data with <ins> 1 cm sampling (0.01m) </ins> the val accuracy was approx. 0.92.

The pipeline is composed of serveral steps and input parametes in `/run_bash_scripts/sem_seg_sean.sh and /run_bash_scripts/tls.h`  which can be set before the run. 

The subset of the default parameters are as follows:
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


## Running semantic segmentation
Semantic segmentation should be run before the instance segmentation since the latter one requires results from the semantic segmentation. 

To run semantic segmentation follow:
```
bash run_bash_scripts/sem_seg_sean.sh -d folder_name
```
Make sure that you put the data in `*.las` or  `*.laz` format to this folder. 

This is a basic run of the command. There are more parameters to be set. Take a look into `run_bash_scripts/sem_seg_sean.sh` to check them.

## Running instance segmentation
To run instance segmentation follow:

```
bash run_bash_scripts/tls.sh -d folder_name

```

This is a basic run of the command. There are more parameters to be set. Take a look into `run_bash_scripts/tls.sh` to check them.

# The stages of the steps executed in the pipeline are as follows :
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

# The paper
[Maciej Wielgosz, Stefano Puliti, Phil Wilkes, Rasmus Astrup. (2023). Point2Tree(P2T). arXiv preprint. arXiv:2305.02651.](https://arxiv.org/abs/2305.02651)


# Orginal repo
For the orignal repo, please take a look there: 

https://github.com/philwilkes/FSCT

https://github.com/philwilkes/TLS2trees




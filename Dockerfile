FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

# install conda
ARG UBUNTU_VER=20.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sudo \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    libtiff5 \
    && rm -rf /var/lib/apt/lists/* 


RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh" && \
    bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b && \
    rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh 

RUN /miniconda/bin/conda update conda 

RUN /miniconda/bin/conda init bash
RUN /miniconda/bin/conda create --name pdal-env python=3.8.13

SHELL ["/miniconda/bin/conda", "run", "-n", "pdal-env", "/bin/bash", "-c"]

RUN echo "conda activate pdal-env" >> ~/.bashrc

RUN conda install -c conda-forge pdal==2.4.3 python-pdal==3.1.2

RUN pip install parse oci ocifs

COPY requirements.txt app/requirements.txt
RUN pip install --no-cache -r app/requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["/miniconda/bin/conda", "run", "-n", "pdal-env", "python", "/app/run_oracle_wrapper.py"]




FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

RUN ln -fs /usr/share/zoneinfo/Europe/Warsaw /etc/localtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    language-pack-en-base \
    openssh-server \
    openssh-client \
    python3.6 \
    python3-pip \
    python3-setuptools \
    ssh \
    curl \
    sudo \
    vim \
    wget \
    less \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade \
    autopep8 \
    doc8 \
    docutils \
    ipython \
    pip \
    pylint \
    pytest \
    rope \
    setuptools \
    wheel \
    torch \
    tqdm \
    pandas
    
# Create non-root user
ARG UID=1000
ARG GID=1000
ARG USERNAME=nibio
ENV HOME /home/${USERNAME}

RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -ms /bin/bash -u ${UID} -g ${GID} -G sudo ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd 

COPY ./check_print.py $HOME

#USER $USERNAME
WORKDIR $HOME

# install conda
ARG UBUNTU_VER=20.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64

RUN mkdir conda_installation && cd conda_installation
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN /miniconda/bin/conda update conda 

RUN /miniconda/bin/conda init bash
RUN /miniconda/bin/conda create --name pdal-env python=3.8.13

SHELL ["/miniconda/bin/conda", "run", "-n", "pdal-env", "/bin/bash", "-c"]

RUN echo "conda activate pdal-env" >> ~/.bashrc

RUN conda install -c conda-forge pdal python-pdal

#CMD ["echo", "'just test print bash'" ]

CMD ["python3", "check_print.py" ]

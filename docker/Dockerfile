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
    
# OpenSSH configuration
RUN mkdir /var/run/sshd \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
                                                    
# Create non-root user
ARG UID=1000
ARG GID=1000
ARG USERNAME=nibio
ENV HOME /home/${USERNAME}

RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -ms /bin/bash -u ${UID} -g ${GID} -G sudo ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && mkdir -p ${HOME}/data ${HOME}/.vscode-server ${HOME}/.ssh \
    && chown ${USERNAME}:${USERNAME} ${HOME}/data ${HOME}/.vscode-server ${HOME}/.ssh


VOLUME [ "${HOME}/data", "${HOME}/.ssh", "${HOME}/.vscode-server"]

# Expose SSH
EXPOSE 22

CMD [ "/usr/sbin/sshd", "-D" ]

FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# Install any python packages you need
COPY requirements.txt requirements.txt
# VOLUME . /GEM

RUN python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch==2.0.1 torchvision

# Set the working directory
WORKDIR /GEM

# Set the entrypoint
# ENTRYPOINT [ "python3" ]

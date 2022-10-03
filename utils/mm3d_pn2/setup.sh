#!/bin/bash

CUDA_HOME=/usr/local/cuda-10.1 pip install mmcv-full
CUDA_HOME=/usr/local/cuda-10.1 pip install git+https://github.com/open-mmlab/mmdetection.git
CUDA_HOME=/usr/local/cuda-10.1 pip install git+https://github.com/open-mmlab/mmsegmentation.git
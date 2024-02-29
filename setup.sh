#!/bin/bash

############## make sure you can write in \tmp; Or you should set TORCH_EXTENSIONS_DIR
# e.g. export TORCH_EXTENSIONS_DIR=/mnt/lustre/$YourUserName$/tmp

# setup completion

#cd completion
#pip install -r requirements.txt


cd utils/mm3d_pn2/
sh setup.sh

############## make sure NVCC is in your environment
# SLURM users
# sh run_build.sh
# or
pip install -v -e .
# python setup.py develop

cd ../../

# setup registration

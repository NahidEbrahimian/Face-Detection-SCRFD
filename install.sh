#!bin/bash

python3.8 -m venv env-scrfd

source env-scrfd/bin/activate

# First you need to Upgrade pip
env-scrfd/bin/pip install --upgrade pip
env-scrfd/bin/pip install torch==1.5.1 torchvision==0.6.1

env-scrfd/bin/pip install -U openmim
env-scrfd/bin/mim install mmcv-full==1.3.3
env-scrfd/bin/mim install mmcv==1.3.3

# git clone https://github.com/open-mmlab/mmdetection.git
cd ./mmdetection
./env-scrfd/bin/python setup.py install # change to your absolute path: ./env-scrfd/bin/python

cd ../
env-scrfd/bin/pip install -r ./requirements/build.txt
./env-scrfd/bin/python setup.py install # change to your absolute path: ./env-scrfd/bin/python  # or "python setup.py develop" 

env-scrfd/bin/pip uninstall pycocotools
echo "Y"
env-scrfd/bin/pip install mmpycocotools
env-scrfd/bin/pip uninstall mmpycocotools
echo "Y"
env-scrfd/bin/pip install mmpycocotools

env-scrfd/bin/pip install numpy==1.23.5
env-scrfd/bin/pip install scipy


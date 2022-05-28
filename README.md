# General_detection

This is a generic object detection project.

## Installation
### 1. Clone this repo
```
git clone https://github.com/DAMONYLY/General_detection.git
```
### 2. Create environment
```
conda create -n G_detection python=3.8
conda activate G_detection
# install torch and torchvison according to your CUDA version
conda install pytorch torchvision
# Installing dependency libraries
pip3 install -r requirements.txt
```

## Train
Now we only support dataset as COCO format. So first convert the dataset to coco format.
### 1. Run utils/voc2coco.sh to convert datasetc format to COCO format (if using coco data, please ignore)

For more usage refer to [x2coco.py](https://paddledetection.readthedocs.io/tutorials/Custom_DataSet.html)

```
sh voc2coco.sh
```
### 2. Setting the config file and run

Change the parameters in config/xx.yaml. More information can be found here 
```
# Change the parameters in config/xx.yaml
# Run train.py
python train.py
```
---
---
## Reference
* This repository is mainly modified from this repository. https://github.com/Peterisfar/YOLOV3


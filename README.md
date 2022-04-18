# YOLOV3
---
# Introduction
This is the YOLOV3 project written in pytorch. 

Now this repo only support the PASCAL VOC dataset.

Subsequently, i will continue to update the code to make it more concise , and add the new and efficient tricks.

---
## Results

| name | Train Dataset | Val Dataset | Batchsize | AP_50 | notes |
| :----- | :----- | :------ | :-----| :-----| :-----|
| YOLOV3-320-640 | 2012trainval | 2007test | 40*80epoch(0.15) | 0.5038 | IOU_loss+Focal_loss |
| YOLOV3-320-640 | 2012trainval | 2007test | 40*80epoch(0.15) | 0.4978 | SmoothL1_loss+Focal_loss |

`Note` : 

* YOLOV3-320-640 means train and test image size is random from 320 to 640. 


## Install
### 1. Clone this repo
```
git clone https://github.com/DAMONYLY/General_detection.git
```
### 2. Download VOC data

Save it in the following form
```
|-- data
    |-- VOCdevkit
        |-- VOC2007
            |-- Annotations
            |-- ImageSets
            |-- JPEGImages
            |-- SegmentationClass
            |-- SegmentationObject
        |-- VOC2007_test
            |-- Annotations
            |-- ImageSets
            |-- JPEGImages
            |-- SegmentationClass
            |-- SegmentationObject
        |-- VOC2012
            |-- Annotations
            |-- ImageSets
            |-- JPEGImages
            |-- SegmentationClass
            |-- SegmentationObject                
```

### 3. Run voc.py to convert data format
Convert data format : Convert the pascal voc *.xml format to custom format (Image_path0   xmin0,ymin0,xmax0,ymax0,class0   xmin1,ymin1...)

Change the DATA_PATH in voc.py to your own voc data_path. like ` DATA_PATH = "{own_path}/General_detection/data/" `
```
python voc.py
```
### 4. Download pretrain weight file
* Darknet pre-trained weight :  [darknet53-448.weights](https://pjreddie.com/media/files/darknet53_448.weights) 
* This repository test weight : [best.pt](https://pan.baidu.com/s/1MdE2zfIND9NYd9mWytMX8g)

Make dir `weight/` and put the weight file in.

## Train
```
# Change the parameters in cfg
# Run train.py
python train.py
```
---
## TODO

* [ ] Mish
* [ ] OctConv
* [ ] Custom data
* [ ] ATSS


---
## Reference
* This repository is mainly modified from this repository. https://github.com/Peterisfar/YOLOV3


# YOLOV3
---
# Introduction
This is the YOLOV3 project written in pytorch. 

Now this repo only support the PASCAL VOC dataset.

Subsequently, i will continue to update the code to make it more concise , and add the new and efficient tricks.

---
## Results

### Origin repo results
| name | Train Dataset | Val Dataset | mAP(others) | mAP(mine) | notes |
| :----- | :----- | :------ | :----- | :-----| :-----|
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 0.769 | 0.768 \| - | baseline(augument + step lr) |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.793 | 0.803 \| - | \+multi-scale training |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.806 | 0.811 \| - | \+focal loss(note the conf_loss in the start is lower) |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.808 | 0.813 \| - | \+giou loss |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.812 | 0.821 \| - | \+label smooth |  
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.822 | 0.826 \| - | \+mixup |  
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.833 | 0.832 \| 0.840 | \+cosine lr |
| YOLOV3-\*-* | 2007trainval + 2012trainval | 2007test | 0.858 | 0.858 \| 0.860 | \+multi-scale test and flip, nms threshold is 0.45 |  

### This repo results
| name | Train Dataset | Val Dataset | BatchSize | mAP | notes |
| :----- | :----- | :------ | :-----| :-----| :-----|
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 15*50epoch | 0.780085 | baseline |
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 15*50epoch | 0.805921 | + focal | 
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 15*50epoch | 0.778626 | + GIOU |
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 15*50epoch | 0.798533 | + focal, GIOU |
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 15*50epoch | 0.803149 | + focal, DIOU | 
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 15*50epoch | 0.802956 | + focal, CIOU |
| YOLOV3-* -544 | 2007trainval + 2012trainval | 2007test | 10* 50epoch | 0.827054 | + Multi, focal, CIOU | 

`Note` : 

* YOLOV3-448-544 means train image size is 448 and test image size is 544. `"*"` means the multi-scale.
* mAP(mine)'s format is (use_difficult mAP | no_difficult mAP).
* In the test, the nms threshold is 0.5(except the last one) and the conf_score is 0.01.`others` nms threshold is 0.45(0.45 will increase the mAP)
* Now only support the single gpu to train and test.

## How to use
### 1. Clone this repo
```
git clone https://github.com/DAMONYLY/General_detection.git
```

### 2. Run utils/voc2coco.sh to convert VOC data format to COCO format (if using coco data, please ignore)

For more usage refer to [x2coco.py](https://paddledetection.readthedocs.io/tutorials/Custom_DataSet.html)

```
sh voc2coco.sh
```

## Train
```
# Change the parameters in config/xx.yaml
# Run train.py
python train.py
```
---
---
## Reference
* This repository is mainly modified from this repository. https://github.com/Peterisfar/YOLOV3


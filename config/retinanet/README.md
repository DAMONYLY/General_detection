# Retinanet

## Model Zoo

### Retinanet on Pasacl VOC
**Notes:**
- All following models are trained on VOC_train07+12 with 2 GPUs(RTX 3080 Ti) and evaludated on VOC_test07. 
- 0.01_1x means using the SGD optimizer with lr 0.01 to train 20 epochs, the learning rate is multiplied by 0.1 at [15,18] epochs respectively.
- Box AP= `mAP(IoU=0.50)`.

| backbone | input_size | batchsize per GPU | training strategy | Box AP | config | log |
| :------: | :--------: | :---------------: | :---------------: | :----: | :----: | :-: |
| ResNet50 | 640        |    10             |   0.01_1x         | 0.741  |  [config](https://github.com/DAMONYLY/General_detection/blob/main/config/retinanet/retinanet_r50_fpn_1x_voc2coco.yaml)  |   [log](https://github.com/DAMONYLY/General_detection/blob/main/config/retinanet/retinanet_r50_fpn_1x_voc2coco_train.log)   | 

## Citations
```latex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```

# coding=utf-8
# project
from config.yolov3_config_voc import MODEL


DATA_PATH = "/raid/yaoliangyong/General_detection/dataset"
PROJECT_PATH = "/raid/yaoliangyong/General_detection"


DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        "NUM":20}
        
# model
MODEL = {
    "name": 'General', # [YOLO, General]
    "backbone": 'Darknet53', # [Darknet53, Resnet]
    "fpn": 'fpn',
    "out_stride": [8, 16, 32], # TODO: 用于输出多少x的特征图用来检测
    'head': 'normal',
    "ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj
    "STRIDES":[8, 16, 32],
    "ANCHORS_PER_SCLAE":9,
    "metrics": 'yolo',
    "loss": 'fun',
         }

# train
TRAIN = {
         "TRAIN_IMG_SIZE":448,
         "AUGMENT":True,
         "BATCH_SIZE":10,
         "MULTI_SCALE_TRAIN":True,
         "IOU_THRESHOLD_LOSS":0.5,
         "EPOCHS":50,
         "NUMBER_WORKERS":4,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1e-3,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":2  # or None
         }


# test
TEST = {
        "TEST_IMG_SIZE":544,
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":0,
        "CONF_THRESH":0.01,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False
        }

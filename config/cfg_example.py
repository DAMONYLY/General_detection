# coding=utf-8
# project


DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        "NUM":20}
        
# model
MODEL = {
    "name": 'General', # [YOLO, General]
    "backbone": 'Resnet', # [Darknet53, Resnet]
    'depth': 18, # depth of ResNet model [18, 34, 50, 101, 152]
    "fpn": 'fpn',
    "out_stride": [8, 16, 32], # TODO: 用于输出多少x的特征图用来检测
    'head': 'normal',
    "ANCHORS":[[[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]],  # Anchors for big obj
            [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],  # Anchors for medium obj
            [[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]]] ,# Anchors for small obj
    "STRIDES":[8, 16, 32],
    "ANCHORS_PER_SCLAE":3,
    "metrics": 'yolo',
    "loss": 'fun',
         }

# train
TRAIN = {
         "TRAIN_IMG_SIZE":320,
         "AUGMENT":False,
         "MULTI_SCALE_TRAIN":False,
         "IOU_THRESHOLD_LOSS":0.5,
         "EPOCHS":151,
         "NUMBER_WORKERS":4,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1e-3,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":0  # or None
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

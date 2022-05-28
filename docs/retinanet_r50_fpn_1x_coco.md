# Model parameter configuration tutorial

Take `retinanet_r50_fpn_1x_coco.yaml` as an example.


```yaml

Classes:
  # Categories in the dataset
  name: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
  # The number of categories in the dataset
  num: 20
    
Model:
  # Name of detector
  name: General # [General]
  backbone:
    # whether to use pretrain model
    pretrain: True 
    # if exist, means load this weight path to pretrain
    weight_path: 
    # backbone type
    name: Resnet # [Darknet53, Resnet, shufflenetv2]
    # depth of ResNet model
    depth: 50 # [18, 34, 50, 101, 152]

  neck:
    # neck type
    name: fpn
    # fpn output channel size, default is 256
    channel_out: 256 
    # TODO: for outputting a feature map of how many x's to use for detection
    out_stride: [8, 16, 32, 64, 128] 
  
  head:
    # head type
    name: retina_head
    # for regression
    reg_head:
      # out number
      out_channel: 4
      # mid conv channel
      mid_channel: 256
      # The number of layers of convolution
      stack_layers: 4
    # for classification
    cls_head:
      # out number
      out_channel: 20
      # mid conv channel
      mid_channel: 256
      # The number of layers of convolution
      stack_layers: 4
    # TODO: Used to specify whether the head is shared for different neck out layers
    head_shared: True

  anchors: 
    # anchor corresponds to the size of the original image, same as the number of neck out layer
    strides: [8, 16, 32, 64, 128]
    # Aspect Ratio
    ratios: [0.5, 1, 2]
    # Area scaling factor
    scales: [4, 5, 6]
    # Number of anchors in a point, should be equal to len(ratios) * len(scales)
    num: 9

  metrics: 
    # assign type
    name: Max_iou # [Max_iou, ATSS]
    # Positive sample iou threshold
    pos_iou_thr: 0.5
    # Negative sample iou threshold
    neg_iou_thr: 0.4

  sample:
    # sample type
    name: Retinanet
    # bbox_coder type
    bbox_coder: bbox2delta

  loss:
    # loss type
    name : Reg_Cls_Loss
    # loss for regression
    reg_loss: 
      # regression loss type
      name: SmoothL1_Loss #[SmoothL1_Loss, IOU_Loss]
      # regression loss ratio
      ratio: 10
    # loss for classification
    cls_loss: 
      # classification loss type
      name: Focal_Loss # [Focal_Loss, BCE_Loss]
      # classification loss ratio
      ratio: 1

Data:
  # type of dataset, now only support coco
  dataset_type: coco 
  # for train
  train:
    # path of train dataset
    dataset_path: ./dataset/voc2coco 
    # name of train dataset
    set_name: train2017
    # data augment
    pipeline:
      # resize
      input_size: [640, 640]
      # whether to keep aspect ratio
      keep_ratio: False
      # hsv
      hsv_prob: 1.0
      # flip
      flip_prob: 0.5
  # for test
  test:
    # path of test dataset
    dataset_path: ./dataset/voc2coco 
    # name of test dataset
    set_name: val2017
    # data augment
    pipeline:
      # resize
      input_size: [640, 640]
      # whether to keep aspect ratio
      keep_ratio: False
    # nms threshold
    nms_thre: 0.5
      
Schedule:
  # if exist, means resume training from resume path
  resume_path: 
  device:
    # cuda device, i.e. 0 or 0,1,2,3 now not support spu
    gpus: '1' 
    # batch size
    batch_size: 10
    # num worker
    num_workers: 4
  # experiments seed
  seed: 
  # train epochs
  epochs: 30

  optimizer:
    # optimizer type
    name: SGD
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.0001
  lr_schedule:
    # lr schedule type
    name: MultiStep_LR
    milestones: [15,25]
    gamma: 0.1
    warmup: 0.5

Log:
  # where to save model(.pth) and tensorboard data
  save_path: ./results/test5 
  # whether to use tensorboard to collect results
  tensorboard: True 
  # epochs interval to val
  val_intervals: 1  

```
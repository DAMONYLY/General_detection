# coding=utf-8
import os
import sys
from time import daylight
sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
import config.cfg_example as cfg
import cv2
import numpy as np
import random

import utils.data_augment as dataAug
import utils.tools as tools

DATA_PATH = "/raid/yaoliangyong/General_detection/dataset"
PROJECT_PATH = "/raid/yaoliangyong/General_detection"
class VocDataset(Dataset):
    def __init__(self, anno_file_type, img_size=416):
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        # get txt file, one raw means one pic
        self.__annotations = self.__load_annotations(anno_file_type)
        self.mix = False
        self.label = {}
        #cupy
    def __len__(self):
        return  len(self.__annotations)

    def __getitem__(self, item):
        # get pic and box
        # if item not in self.label:
        img, bboxes = self.__parse_annotation(self.__annotations[item])
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        if self.mix:
            item_mix = random.randint(0, len(self.__annotations)-1)
            img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
            img_mix = img_mix.transpose(2, 0, 1)

            img, bboxes = dataAug.Mixup()(img, bboxes, img_mix, bboxes_mix)
        else:
            img, bboxes = dataAug.Mixup(p=2)(img, bboxes)
        # nl = len(bboxes)
        # labels_out = torch.zeros((nl, 6))
        
        # if nl:
            # labels_out[:, :] = torch.from_numpy(bboxes)
            # self.label[item] = [torch.from_numpy(img).float(), labels_out]
        return torch.from_numpy(img).float(), torch.from_numpy(bboxes)
        # else:
            # return self.label[item][0], self.label[item][1]
        
        
    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, -1] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)
    
    @staticmethod
    def collate_fn_batch(batch):
        imgs, labels = zip(*batch)  # transposed
        batchsize = len(labels)
        max_num_labels = max(label.shape[0] for label in labels)
        
        if max_num_labels > 0:
            label_padded = torch.ones((batchsize, max_num_labels, 6)) * -1
            for idx, label in enumerate(labels):
                label_padded[idx, :label.shape[0]] = label
        else:
            label_padded = torch.ones((batchsize, 1, 6)) * -1

        return torch.stack(imgs, 0), label_padded

    def __load_annotations(self, anno_type):

        assert anno_type in ['train', 'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(PROJECT_PATH, 'data', anno_type+"_annotation.txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x:len(x)>0, f.readlines()))
        assert len(annotations)>0, "No images found in {}".format(anno_path)

        return annotations

    def __parse_annotation(self, annotation):
        """从txt文件中提取图片路径以及boxes.
        Args:
            annotation (str): txt文件中的某一行
        Returns:
            img: 图片
            bboxes：lable的box，[x1,y1,x2,y2,cls_id]

        """
        anno = annotation.strip().split(' ')

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        # augment
        # img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        # img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        # img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        
        return img, bboxes

    def __creat_label(self, bboxes):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """

        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6+self.num_classes))
                                                                      for i in range(3)]
        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]   # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("bbox_xywh: ", bbox_xywh)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                # 中心点向下取整，即到左上角，再加上0.5，移动到grid中心
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        # [56, 56, 3, 26], [28, 28, 3, 26], [14, 14, 3, 26], [150, 4], [150, 4], [150, 4]
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == "__main__":

    voc_dataset = VocDataset(anno_file_type="train", img_size=448)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)

    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i==0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)

            if img.shape[0] == 1:
                labels = np.concatenate([label_sbbox.reshape(-1, 26), label_mbbox.reshape(-1, 26),
                                         label_lbbox.reshape(-1, 26)], axis=0)
                labels_mask = labels[..., 4]>0
                labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
                                        axis=-1).reshape(-1, 1)], axis=-1)

                print(labels.shape)
                tools.plot_box(labels, img, id=1)

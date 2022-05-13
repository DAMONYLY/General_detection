python x2coco.py \
        --dataset_type voc \
        --voc_anno_dir /raid/yly/General_detection/dataset/VOCdevkit/VOC2012/Annotations/ \
        --voc_anno_list /raid/yly/General_detection/dataset/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt \
        --voc_label_list ./label_list.txt \
        --voc_out_name test.json
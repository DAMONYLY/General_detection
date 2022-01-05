python x2coco.py \
        --dataset_type voc \
        --voc_anno_dir dataset/VOCdevkit/VOC2007_test/Annotations/ \
        --voc_anno_list dataset/VOCdevkit/VOC2007_test/ImageSets/Main/test.txt \
        --voc_label_list dataset/VOCdevkit/label_list.txt \
        --voc_out_name voc_val.json
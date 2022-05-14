import os
import cv2
import numpy as np
from .draw_on_pic import visualize_boxes

def show_dataset(dataloader, path, num=10):
    count = 0
    if not os.path.exists(path):
        os.mkdir(path)
    for i, data in enumerate(dataloader):
        imgs = data['imgs']
        bboxes = data['targets']
        batch = imgs.size()[0]
        vis_imgs = imgs.numpy()
        vis_boxes = bboxes.numpy()
        for i in range(batch):
            vis_img = vis_imgs[i].transpose(1,2,0)
            img_names = os.path.join(path, "ori"+ str(count) + '.jpg')
            cv2.imwrite(img_names, vis_img)
            vis_box = vis_boxes[i]
            for item in range(vis_box.shape[0]):
                if vis_box[item][-1] == -1:
                    vis_box = vis_box[:item]
                    break
            _labels = vis_box[:, -1].astype(np.int32)
            _probs = np.ones_like(_labels)
            cateNames = dataloader.dataset.class_names
            visualize_boxes(image=vis_img, boxes=vis_box[:, :-1], labels=_labels, 
                            probs=_probs, class_labels=cateNames)
            img_names_with_bbox = os.path.join(path, 'bbox_' + "ori"+ str(count) + '.jpg')
            cv2.imwrite(img_names_with_bbox, vis_img)
            print(f'saved to {img_names_with_bbox}')
            count += 1
            if count == num:
                return
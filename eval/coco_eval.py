from pycocotools.cocoeval import COCOeval
import json
import torch
from tqdm import tqdm
import os
from model.anchor import Anchors
from model.post_processing import yolo_decode, clip_bboxes, multiclass_nms

class COCO_Evaluater:
    '''
    COCO evaluate
    '''
    def __init__(self, dataloader, device, args, conf_thre=0.05):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.device = device
        self.img_size = args.Data.test.pipeline.input_size
        self.conf_thre = conf_thre
        self.nms_thre = args.Data.test.nms_thre
        self.anchor = Anchors(args.Model.anchors)
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]
    
    def evalute(self, model, save_path='./'):
        model = model.eval()
        data_list = []     
        eval_results = {}   
        for i, data in enumerate(tqdm(self.dataloader)):
            with torch.no_grad():
                imgs = data['imgs'].to(self.device)
                outputs = model(imgs)
                batch_boxes, batch_scores, batch_labels = self.postprocess(imgs, 
                                                                           self.anchor,
                                                                           outputs[0],
                                                                           outputs[1])
                data_list.extend(self.convert_to_pycocotools(data, batch_boxes, 
                                                             batch_scores, batch_labels))
        if not len(data_list):
            return 

        # write output
        path = os.path.join(save_path, '{}_bbox_results.json'.format(self.dataset.set_name))
        json.dump(data_list, open(path, 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = self.dataset.coco
        coco_pred = coco_true.loadRes(path)

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        aps = coco_eval.stats[:6]
        model.train()

        
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v

        return eval_results
    

    def convert_to_pycocotools(self, data, batch_boxes, batch_scores, batch_labels):
        assert len(batch_boxes) == len(batch_scores) == len(batch_labels)
        batch_size = len(batch_boxes)
        results = []
        for batch in range(batch_size):
            boxes  = batch_boxes[batch].cpu()
            scores = batch_scores[batch].cpu()
            labels = batch_labels[batch].cpu()

            # correct boxes for image scale
            boxes /= data['info'][batch]['scale']
            index = data['info'][batch]['img_id']

            if boxes.shape[0] == 0:
                continue
            # change to (x, y, w, h) (MS COCO standard)
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

            # compute predicted labels and scores
            #for box, score, label in zip(boxes[0], scores[0], labels[0]):
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]

                # scores are sorted, so we can break
                if score < self.conf_thre:
                    break

                # append detection for each positively labeled class
                image_result = {
                    'image_id'    : index,
                    'category_id' : self.dataset.label_to_coco_label(label),
                    'score'       : float(score),
                    'bbox'        : box.tolist(),
                }

                # append detection to results
                results.append(image_result)
        return results
    
    
    def postprocess(self, imgs, anchor_generater, proposals_regs, proposals_clses):
        anchors = anchor_generater(imgs)
        batch_boxes, batch_scores, batch_labels = [], [], []
        batch_size, _, h, w = imgs.shape
        for id in range(batch_size):
            proposals_reg = proposals_regs[id]
            proposals_cls = proposals_clses[id]
            p_reg = yolo_decode(proposals_reg, anchors)
            p_cls = proposals_cls.squeeze(0)
            
            # 2. 将超出图片边界的框截掉
            p_reg = clip_bboxes(p_reg, imgs)
            padding = p_cls.new_zeros(p_cls.shape[0], 1)
            p_cls = torch.cat([p_cls, padding], dim=1)
            # scores, labels, boxes = nms_boxes(p_reg, p_cls)
            boxes, scores, labels = multiclass_nms(
                multi_bboxes = p_reg,
                multi_scores = p_cls,
                score_thr=self.conf_thre,
                nms_cfg=dict(type="nms", iou_threshold=self.nms_thre),
                max_num=100,
            )
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_labels.append(labels)
        return batch_boxes, batch_scores, batch_labels
        


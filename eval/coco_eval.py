from cProfile import label
from pycocotools.cocoeval import COCOeval
import torchvision
import json
import torch
from tqdm import tqdm
import os
from model.anchor import Anchors
from model.post_processing import yolo_decode, clip_bboxes, multiclass_nms
from model.utils import gather, is_main_process
import itertools
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
        progress_bar = tqdm if is_main_process() else iter
        for i, data in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = data['imgs'].to(self.device)
                outputs = model(imgs)
                proposals_regs = torch.cat(outputs[0], dim=1)
                proposals_clses = torch.cat(outputs[1], dim=1)
                output = self.postprocess(self.anchor, proposals_regs, proposals_clses)
                data_list.extend(self.convert_to_pycocotools(data, output))
        data_list = gather(data_list, dst=0)
        data_list = list(itertools.chain(*data_list))
        if not is_main_process():
            return 0

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

    def postprocess(self, anchor_generater, proposals_regs, proposals_clses):

        anchors = anchor_generater(self.img_size, self.device, proposals_regs.dtype)

        assert len(proposals_regs) == len(proposals_clses)
        batch_size = len(proposals_regs)
        output = [None for _ in range(batch_size)]
        for id in range(batch_size):
            proposals_reg = proposals_regs[id]
            proposals_cls = proposals_clses[id]

            if proposals_reg.size(1) > 4:
                p_obj = proposals_reg[:, 4]
            else:
                p_obj = proposals_reg.new_full((proposals_reg.size(0), ), 1)
            
            p_reg = proposals_reg[:, :4]
            p_reg = yolo_decode(p_reg, anchors)
            p_reg = clip_bboxes(p_reg, self.img_size)

            class_conf, class_pred = torch.max(proposals_cls, 1, keepdim=True)

            conf_mask = (p_obj * class_conf.squeeze() >= self.conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((p_reg, p_obj.unsqueeze(1), class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]

            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                self.nms_thre,
            )

            detections = detections[nms_out_index]
            if output[id] is None:
                output[id] = detections
            else:
                output[id] = torch.cat((output[id], detections))

        return output

    def convert_to_pycocotools(self, data, outputs):
        
        results = []
        for batch, output in enumerate(outputs):
            # output = output
            if output is None:
                continue
            boxes  = output[:, :4].cpu()
            scores = (output[:, 4] * output[:, 5]).cpu()
            labels = output[:, 6].cpu()

            # correct boxes for image scale
            boxes /= data['info'][batch]['scale']
            index = data['info'][batch]['img_id']

            if boxes.shape[0] == 0:
                continue
            # change to (x, y, w, h) (MS COCO standard)
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

            # compute predicted labels and scores
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id].numpy().item())
                label = int(labels[box_id])
                box = boxes[box_id, :].numpy()

                # append detection for each positively labeled class
                image_result = {
                    'image_id'    : index,
                    'category_id' : self.dataset.label_to_coco_label(label),
                    'score'       : score,
                    'bbox'        : box.tolist(),
                }

                # append detection to results
                results.append(image_result)
        return results

    
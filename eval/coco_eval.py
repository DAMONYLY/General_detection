from pycocotools.cocoeval import COCOeval
import json
import torch
from tqdm import tqdm
import os

def evaluate_coco(dataset, model, save_path, threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        # for index in tqdm(range(len(dataset))):
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                boxes, scores, labels = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0), 'test')
            else:
                boxes, scores, labels = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0), 'test')
            if isinstance(scores, list):
                scores = scores[0]
                labels = labels[0]
                boxes = boxes[0]
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
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
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            # print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open(os.path.join(save_path, '{}_bbox_results.json'.format(dataset.set_name)), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(os.path.join(save_path, '{}_bbox_results.json'.format(dataset.set_name)))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        aps = coco_eval.stats[:6]
        model.train()
        eval_results = {}
        metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]
        for k, v in zip(metric_names, aps):
            eval_results[k] = v
        # print(eval_results[metric_names[1]])
        return eval_results

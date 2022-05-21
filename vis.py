import numpy as np
import argparse
import os
import cv2

from model.build_model import build_model
from utils.gpu import select_device
import torch
import argparse
from utils.tools import *
from utils.model_info import get_model_info
from utils.config import cfg, load_config
from utils.visualize import *
from model.anchor import Anchors
from model.post_processing import yolo_decode, clip_bboxes, multiclass_nms
from utils.draw_on_pic import visualize_boxes

class Trainer(object):
    def __init__(self, args):
        #----------- 1. get gpu info -----------------------------------------------
        self.device = select_device(args.Schedule.device.gpus)
        self.DP = False
        self.pic = args.pic
        self.img_size = args.Data.test.pipeline.input_size
        self.conf_thre = 0.05
        self.nms_thre = 0.3
        #----------- 3. build model -----------------------------------------------
        self.model = build_model(args).to(self.device)
        self.model_info = get_model_info(self.model, args.Data.test.pipeline.input_size)
        print("Model Summary: {}".format(self.model_info))
        self.anchor = Anchors(args.Model.anchors)
        #------------5. resume training --------------------------------------
        if args.Schedule.resume_path:
            print('=> Loading models from {}'.format(args.Schedule.resume_path))
            chkpt = torch.load(args.Schedule.resume_path, map_location=self.device)
            self.model.load_state_dict(chkpt['model'])
            del chkpt
        else:
            raise ValueError('Must have checkpoints path to load checkpoints')
        #-------------6. DP mode ------------------------------
        if self.device and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(self.model)
            self.model = model.to(self.device)
            self.DP = True

    def vis(self, save_path='./test'):
        with torch.no_grad():
            if os.path.isdir(self.pic):
                pic = os.listdir(self.pic)
                # print(pic)
            elif os.path.isfile(self.pic):
                pic_path = os.path.split(self.pic)
                self.pic = pic_path[0]
                pic = [pic_path[1]]
            for single_pic in pic:
                ori_img = self.load_img(os.path.join(self.pic, single_pic))
                img = torch.from_numpy(ori_img.transpose(2, 0, 1)).unsqueeze(0)
                img = img.to(self.device).type(torch.float32)
                outputs = self.model(img)
                proposals_regs = torch.cat(outputs[0], dim=1)
                proposals_clses = torch.cat(outputs[1], dim=1)
                boxes, scores, labels = self.postprocess(self.anchor, proposals_regs, proposals_clses)
                pos_inds = scores > 0.5
                boxes = boxes[pos_inds].cpu().numpy()
                scores = scores[pos_inds].cpu().numpy()
                labels = labels[pos_inds].cpu().numpy()
                visualize_boxes(image=ori_img, 
                                boxes=boxes, 
                                labels=labels, 
                                probs=scores, class_labels=cfg.Classes.name)
                pic_name = os.path.join(save_path, 'deteced_' + single_pic)
                cv2.imwrite(pic_name, ori_img)
                print(f'saved to {pic_name}')
            

    def postprocess(self, anchor, proposals_regs, proposals_clses):
        anchors = anchor(self.img_size, self.device, proposals_regs.dtype)
        batch_boxes, batch_scores, batch_labels = [], [], []
        assert len(proposals_regs) == len(proposals_clses)
        batch_size = len(proposals_regs)
        for id in range(batch_size):
            proposals_reg = proposals_regs[id]
            proposals_cls = proposals_clses[id]
            p_reg = yolo_decode(proposals_reg, anchors)
            p_cls = proposals_cls.squeeze(0)
            
            # 2. 将超出图片边界的框截掉
            p_reg = clip_bboxes(p_reg, self.img_size)
            
            padding = p_cls.new_zeros(p_cls.shape[0], 1)
            p_cls = torch.cat([p_cls, padding], dim=1)

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
        return batch_boxes[0], batch_scores[0], batch_labels[0]

	
    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, 
                         (int(self.img_size[0]), int(self.img_size[1])),
                         interpolation=cv2.INTER_LINEAR,
                         ).astype(np.uint8)
        return img


if __name__ == "__main__":

    import sys 
    # sys.argv = ['train.py', '--b', '40', '--device', '0' ]
    default_config_parser = parser = argparse.ArgumentParser(description= 'General Detection config parser')
    parser.add_argument('--config', type=str, default='./results/Resnet50_lr0.01_atss/experiments.yaml', help="train config file path")
    parser.add_argument('--model_path', type=str, default='./results/Resnet50_lr0.01_atss/backup_epoch29.pt', help="model checkpoints")
    parser.add_argument('--pic', type=str, default='./dataset/voc2coco/val2017', help="pic path")
    opt = parser.parse_args()
    load_config(cfg, opt.config, save=False)
    cfg.defrost()
    cfg.Schedule.resume_path = opt.model_path
    cfg.pic = opt.pic
    cfg.freeze()
    Trainer(cfg).vis()


# def main(args=None):
#     default_config_parser = parser = argparse.ArgumentParser(description= 'General Detection config parser')
#     parser.add_argument('--config', type=str, default='./config/test.yaml', help="train config file path")
#     opt = parser.parse_args()
#     load_config(cfg, opt.config, save=True)

# 	#  -------------- 1. get GPU -----------------------
# 	device = select_device(cfg.Schedule.device.gpus)

# 	#---------------- 2. build model -----------------------------------------------
# 	model = build(cfg).to(device)
# 	model_info = get_model_info(model, cfg.TEST['TEST_IMG_SIZE'])
# 	print("Model Summary: {}".format(model_info))

# 	#---------------- 3. load resume file --------------------------------------
# 	if args.resume_path:
# 		print('Start load file from {}'.format(args.resume_path))
# 		chkpt = torch.load(args.resume_path, map_location=device)
# 		model.load_state_dict(chkpt['model'])
# 		del chkpt

# 	# --------------- 4. DP mode ---------------------------------------
# 	if device and torch.cuda.device_count() > 1:
# 		model = torch.nn.DataParallel(model)
# 		model = model.to(device)
# 		DP = True
# 	# --------------- 5 Get vla datasets ---------------------------
# 	if args.dataset == 'coco':
# 		dataset_val = CocoDataset(args.dataset_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
# 	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
# 	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

# 	# --------------- 5 Start inference ----------------
# 	model.eval()

# 	unnormalize = UnNormalizer()

# 	def draw_caption(image, box, caption):

# 		b = np.array(box).astype(int)
# 		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
# 		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

# 	for idx, data in enumerate(dataloader_val):
# 		if idx == args.num_pic:
# 			break
# 		with torch.no_grad():
# 			st = time.time()
# 			transformed_anchors, scores, labels = model(data['img'].to(device), 'test')
# 			print('Elapsed time: {}'.format(time.time()-st))
# 			if isinstance(scores, list):
# 				scores = scores[0]
# 				labels = labels[0]
# 				transformed_anchors = transformed_anchors[0]
# 			scores = scores.cpu()
# 			labels = labels.cpu()
# 			transformed_anchors  = transformed_anchors.cpu()

# 			idxs = np.where(scores>0.5)
# 			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

# 			img[img<0] = 0
# 			img[img>255] = 255

# 			img = np.transpose(img, (1, 2, 0))

# 			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

# 			boxes = transformed_anchors[idxs[0]]
# 			class_inds = labels[idxs[0]]
# 			scores_box = scores[idxs[0]]
# 			img_2 = deepcopy(img)
# 			visualize_boxes(image=img_2, boxes=boxes, labels=class_inds, probs=scores_box, class_labels=cfg.DATA["CLASSES"])

# 			for j in range(idxs[0].shape[0]):
# 				bbox = transformed_anchors[idxs[0][j], :]
# 				x1 = int(bbox[0])
# 				y1 = int(bbox[1])
# 				x2 = int(bbox[2])
# 				y2 = int(bbox[3])
# 				label_name = dataset_val.labels[int(labels[idxs[0][j]])]
# 				draw_caption(img, (x1, y1, x2, y2), label_name)

# 				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
# 				print(label_name)

# 			path = os.path.join(args.save_path, 'pic/')
# 			if not os.path.exists(path):
# 				os.mkdir(path)
# 			cv2.imwrite(os.path.join(path, "{}_2.jpg".format(idx)), img_2)
# 			print("Num: {}, saved images : {}".format(idx + 1, path))
# 			cv2.imwrite(os.path.join(path, "{}.jpg".format(idx)), img)
# 			print("Num: {}, saved images : {}".format(idx + 1, path))



# if __name__ == '__main__':

# 	import sys 
# 	sys.argv = ['vis.py', '--resume_path', './results/resnet18_warm11/backup_epoch90.pt', '--device', '6', '--save_path', './results/resnet18_warm11/']
# 	main()

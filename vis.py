import numpy as np
import time
import time
import argparse
import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from utils.coco_dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from model.build_model import build
import utils.gpu as gpu
from utils.model_info import get_model_info
import config.cfg_example as cfg



def main(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='coco', help='dataset type')
	parser.add_argument('--dataset_path', type=str, default='./dataset/voc2coco', help='path of dataset')
	parser.add_argument('--resume_path', type=str, default='', help='path of model file to resume')
	parser.add_argument('--save_path', type=str, default='', help='save pic path')
	parser.add_argument('--device', default='7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--num_pic', type=int, default=100, help='max numbers of saving pic')


	args = parser.parse_args()

	#  -------------- 1. get GPU -----------------------
	device = gpu.select_device(args.device)

	#---------------- 2. build model -----------------------------------------------
	model = build(cfg).to(device)
	model_info = get_model_info(model, cfg.TEST['TEST_IMG_SIZE'])
	print("Model Summary: {}".format(model_info))

	#---------------- 3. load resume file --------------------------------------
	if args.resume_path:
		print('Start load file from {}'.format(args.resume_path))
		chkpt = torch.load(args.resume_path, map_location=device)
		model.load_state_dict(chkpt['model'])
		del chkpt

	# --------------- 4. DP mode ---------------------------------------
	if device and torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
		model = model.to(device)
		DP = True
	# --------------- 5 Get vla datasets ---------------------------
	if args.dataset == 'coco':
		dataset_val = CocoDataset(args.dataset_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	# --------------- 5 Start inference ----------------
	model.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):
		if idx == args.num_pic:
			break
		with torch.no_grad():
			st = time.time()
			transformed_anchors, scores, labels = model(data['img'].to(device), 'test')
			print('Elapsed time: {}'.format(time.time()-st))
			if isinstance(scores, list):
				scores = scores[0]
				labels = labels[0]
				transformed_anchors = transformed_anchors[0]
			scores = scores.cpu()
			labels = labels.cpu()
			transformed_anchors  = transformed_anchors.cpu()

			idxs = np.where(scores>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(labels[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)

			path = os.path.join(args.save_path, 'pic/')
			if not os.path.exists(path):
				os.mkdir(path)
			cv2.imwrite(os.path.join(path, "{}.jpg".format(idx)), img)
			print("Num: {}, saved images : {}".format(idx + 1, path))



if __name__ == '__main__':

	import sys 
	sys.argv = ['vis.py', '--resume_path', './results/resnet18_warm5/backup_epoch7.pt', '--device', '4,5', '--save_path', './results/resnet18_warm5/']
	main()

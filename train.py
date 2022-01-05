
from model.loss_calculater import Loss_calculater
from model.build_model import build
import utils.gpu as gpu
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.datasets import VocDataset
import time
import datetime
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
# import config.yolov3_config_voc as cfg
import config.cfg_example as cfg
from utils import cosine_lr_scheduler, model_info
from utils.coco_dataloader import AspectRatioBasedSampler, CocoDataset, Resizer, Augmenter, Normalizer, collater
from eval import coco_eval
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'


class Trainer(object):
    def __init__(self, args, gpu_id):
        #----------- 1. init seed for reproduce -----------------------------------
        init_seeds(10)
        #----------- 2. get gpu info -----------------------------------------------
        self.device = gpu.select_device(gpu_id)

        self.start_epoch = 0
        self.best_mAP = 0.
        self.DP = False
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = args.weight_path
        self.save_path = args.save_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.dataset = args.dataset
        #----------- 3. get train dataset ------------------------------------------
        if self.dataset == 'coco':
            
            self.train_dataset = CocoDataset(args.dataset_path,
                            set_name='train2017',
                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
            self.val_dataset = CocoDataset(args.dataset_path,
                                    set_name='val2017',
                                    transform=transforms.Compose([Normalizer(), Resizer()]))
            sampler = AspectRatioBasedSampler(self.train_dataset, batch_size=args.batch_size, drop_last=False)
            self.train_dataloader = DataLoader(self.train_dataset,
                                            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                            batch_sampler=sampler,
                                            collate_fn=collater
                                            )

        #----------- 4. build model -----------------------------------------------
        self.model = build(cfg).to(self.device)
        
        self.model_info = model_info.get_model_info(self.model, cfg.TEST['TEST_IMG_SIZE'])
        print("Model Summary: {}".format(self.model_info))
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        #------------5. build loss calculater--------------------------------
        self.loss_calculater = Loss_calculater(cfg)
        # self.model.apply(tools.weights_init_normal)

        #------------6. init optimizer, criterion, scheduler, weights-----------------------
        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        
        #------------7. resume training --------------------------------------
        if args.resume_path:
            print('Start resume trainning from {}'.format(args.resume_path))
            self.__load_model_weights(self.resume_path)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))
        #-------------8. DP mode ------------------------------
        if self.device and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(self.model)
            self.model = model.to(self.device)
            self.DP = True


    def __load_model_weights(self, resume_path):
        chkpt = torch.load(resume_path, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])

        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mAP = chkpt['best_mAP']
        del chkpt



    def __save_model_weights(self, path, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(path, "best.pt")
        last_weight = os.path.join(path, 'backup_epoch%g.pt'%epoch)
        if self.DP:
            model = self.model.module
        else:
            model = self.model
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)
        del chkpt

    def train(self):

        print("Train datasets number is : {}".format(len(self.train_dataset)))
        all_iter = (self.epochs - self.start_epoch) * len(self.train_dataloader)
        for epoch in range(self.start_epoch, self.epochs):
            
            self.model.train()
            mloss = torch.zeros(3)
            iter_time = 0
            start_time = time.time()
            # for i, (imgs, bboxes)  in enumerate(self.train_dataloader):
            for i, data in enumerate(self.train_dataloader):
                # torch.cuda.synchronize()
                # self.scheduler.step(len(self.train_dataloader)*epoch + i)
                self.optimizer.zero_grad()
                imgs = data['img']
                bboxes = data['annot']
                imgs = imgs.to(self.device)
                bboxes = bboxes.to(self.device)

                features = self.model(imgs)
                loss, loss_reg, loss_cls = self.loss_calculater(imgs, features, bboxes)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_reg.item(), loss_cls.item(), loss.item()])
                mloss = (mloss * i + loss_items) / (i + 1)

                print_fre = 10
                print_average = False
                # Print batch results
                if i != 0 and i% print_fre==0:
                    iter_time = iter_time/10
                    eta_seconds = (all_iter - (epoch - self.start_epoch) * len(self.train_dataloader) - (i - 1)) * iter_time
                    eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
                    if print_average:
                        line = 'Epoch:[{}|{}], Batch:[{}|{}], iter_time:{:.2f}s, loss_all:{:.2f}, loss_reg:{:.2f}, loss_cls:{:.2f}, lr:{:.2g}'.format(
                            epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, iter_time, mloss[2], mloss[0], mloss[1], self.optimizer.param_groups[0]['lr'])
                    else:
                        line = 'Epoch:[{}|{}], Batch:[{}|{}], iter_time:{:.2f}s, loss_all:{:.2f}, loss_reg:{:.2f}, loss_cls:{:.2f}, lr:{:.2g}'.format(
                            epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, iter_time, loss_items[2], loss_items[0], loss_items[1], self.optimizer.param_groups[0]['lr'])
                    print(line +', ' + eta_str)
                    iter_time = 0

                end_time = time.time()
                iter_time += end_time - start_time
                start_time = time.time()
                # break
            self.scheduler.step(mloss[2])
            mAP = 0
            if self.save_path:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.__save_model_weights(self.save_path, epoch, mAP)
                print('best mAP : %g' % (self.best_mAP))
            if epoch >= 2:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    if self.dataset == 'coco':
                        coco_eval.evaluate_coco(self.val_dataset, self.model)



if __name__ == "__main__":

    import sys 
    # sys.argv = ['train.py', '--b', '20', '--device', '7', '--weight_path', 'darknet53_448.weights' ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='', help='weight file path to pretrain')
    parser.add_argument('--dataset', type=str, default='coco', help='dataset type')
    parser.add_argument('--dataset_path', type=str, default='./dataset/voc2coco', help='path of dataset')
    parser.add_argument('--resume_path', type=str, default='', help='path of model file to resume')
    parser.add_argument('--save_path', type=str, default='', help='save model path')
    parser.add_argument('--pre_train', type=bool, default=True, help='whether to use pre-trained models')
    parser.add_argument('--batch_size', '--b', type=int, default=40,  help='mini batch number')
    parser.add_argument('--device', default='7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--local_rank", type=int, default=0)
    opt = parser.parse_args()
    # update_opt_to_cfg(opt, cfg)
    cfg.pre_train = opt.pre_train
    cfg.weight_path = opt.weight_path
    Trainer(args = opt, gpu_id=opt.device).train()
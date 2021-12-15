
from model.loss_calculater import Loss_calculater
from model.build_model import build
import utils.gpu as gpu
import torch
import torch.optim as optim
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


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'


class Trainer(object):
    def __init__(self, args, weight_path, resume, gpu_id):
        #----------- 1. init seed for reproduce -----------------------------------
        init_seeds(0)
        #----------- 2. get gpu info -----------------------------------------------
        self.device = gpu.select_device(gpu_id)

        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]

        #----------- 3. get train dataset ------------------------------------------
        self.train_dataset = VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        # train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True,
                                           collate_fn=VocDataset.collate_fn
                                           )

        #----------- 4. build model -----------------------------------------------
        self.model = build(cfg).double().to(self.device)
        self.model_info = model_info.get_model_info(self.model, cfg.TEST['TEST_IMG_SIZE'])
        print("Model Summary: {}".format(self.model_info))
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        if self.device and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(self.model)
            self.model = model.to(self.device)
        #------------5. build loss calculater--------------------------------
        self.loss_calculater = Loss_calculater(cfg)
        # self.model.apply(tools.weights_init_normal)

        #------------6. init optimizer, criterion, scheduler, weights-----------------------
        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        if args.pre_train and args.weight_path:
            self.__load_model_weights(weight_path, resume)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))


    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
            chkpt = torch.load(last_weight, map_location=self.device)
            self.model.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.model.load_darknet_weights(weight_path)


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
        del chkpt

    def get_loss(self, loss):
        return loss.mean()

    def train(self):
        # print(self.model)
        print("Train datasets number is : {}".format(len(self.train_dataset)))
        all_iter = (self.epochs - self.start_epoch) * len(self.train_dataloader)
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            mloss = torch.zeros(4)
            iter_time = 0
            for i, (imgs, bboxes)  in enumerate(self.train_dataloader):

                start_time = time.time()

                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                imgs = imgs.to(self.device)
                bboxes = bboxes.to(self.device)

                features = self.model(imgs)

                loss, loss_reg, loss_obj, loss_cls = self.loss_calculater(features, bboxes)
                # for multi-GPU compute loss, calculate average loss
                # loss, loss_reg, loss_conf, loss_cls = self.get_loss(loss), self.get_loss(loss_reg), self.get_loss(loss_conf), self.get_loss(loss_cls)

                self.optimizer.zero_grad()
                loss.backward()
                

                self.optimizer.step()
                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_reg.item(), loss_obj.item(), loss_cls.item(), loss.item()])
                mloss = (mloss * i + loss_items) / (i + 1)
                print_fre = 10
                # Print batch results
                if i != 0 and i% print_fre==0:
                    iter_time = iter_time/10
                    eta_seconds = (all_iter - (epoch - self.start_epoch) * len(self.train_dataloader) - (i - 1)) * iter_time
                    eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
                    line = 'Epoch:[{}|{}], Batch:[{}|{}], iter_time:{:.2f}s, loss_all:{:.2f}, loss_reg:{:.2f}, loss_obj:{:.2f}, loss_cls:{:.2f}, lr:{:.2g}'.format(
                        epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, iter_time, mloss[3], mloss[0],mloss[1], mloss[2], self.optimizer.param_groups[0]['lr'])
                    print(line +', ' + eta_str)
                    iter_time = 0

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1) % print_fre == 0:
                    self.train_dataset.img_size = random.choice(range(10, 20)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))
                end_time = time.time()
                iter_time += end_time - start_time

            mAP = 0
            if epoch >= 20:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(self.model).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    print('mAP:%g'%(mAP))

            #self.__save_model_weights(epoch, mAP)
            #print('best mAP : %g' % (self.best_mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='', help='weight file path')
    parser.add_argument('--pre_train', type=str, default=True, help='whether to use pre-trained models')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--batch_size', '--b', type=int, default=20,  help='mini batch number')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--local_rank", type=int, default=0)
    opt = parser.parse_args()

    Trainer(args = opt, weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.device).train()
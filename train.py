
from yaml import parse
from model.loss_calculater import Loss_calculater
from model.build_model import build
import utils.gpu as gpu
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import datetime
import argparse
from utils.load_config import parse_args_and_yaml, Load_config
from utils.tools import *
from tensorboardX import SummaryWriter
from utils import model_info
from utils.coco_dataloader import AspectRatioBasedSampler, CocoDataset, Resizer, Augmenter, Normalizer, collater
from eval import coco_eval
from utils.optimizer import build_optimizer
from utils.config import cfg, load_config
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'


class Trainer(object):
    def __init__(self, args):
        #----------- 1. init seed for reproduce -----------------------------------
        init_seeds(args.Schedule.seed)
        #----------- 2. get gpu info -----------------------------------------------
        self.device = gpu.select_device(args.Schedule.device.gpus)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.DP = False
        self.epochs = args.Schedule.epochs
        
        # self.weight_path = args.weight_path

        self.save_path = args.Log.save_path
        self.multi_scale_train = args.Train.MULTI_SCALE_TRAIN
        self.dataset = args.Data.dataset_type
        self.val_intervals = args.Log.val_intervals
        self.tensorboard = args.Log.tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'logs'))
        #----------- 3. get train dataset ------------------------------------------
        if self.dataset == 'coco':
            self.train_dataset = CocoDataset(args.Data.train.dataset_path,
                            set_name='train2017',
                            transform=transforms.Compose([Normalizer(), Resizer()]))
            self.val_dataset = CocoDataset(args.Data.test.dataset_path,
                                    set_name='val2017',
                                    transform=transforms.Compose([Normalizer(), Resizer()]))
            train_sampler = AspectRatioBasedSampler(self.train_dataset, 
                                                    batch_size=args.Schedule.device.batch_size, 
                                                    drop_last=False
                                                    )
            self.train_dataloader = DataLoader(self.train_dataset,
                                            num_workers=args.Schedule.device.num_workers,
                                            batch_sampler=train_sampler,
                                            collate_fn=collater
                                            )
            val_sampler = AspectRatioBasedSampler(self.val_dataset, 
                                                batch_size=args.Schedule.device.batch_size, 
                                                drop_last=False
                                                )
            self.val_dataloader = DataLoader(self.val_dataset,
                                            num_workers=args.Schedule.device.num_workers,
                                            batch_sampler=val_sampler,
                                            collate_fn=collater
                                            )

        #----------- 4. build model -----------------------------------------------
        self.model = build(args).to(self.device)
        
        self.model_info = model_info.get_model_info(self.model, args.Test.TEST_IMG_SIZE)
        print("Model Summary: {}".format(self.model_info))
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        #------------5. build loss calculater--------------------------------
        self.loss_calculater = Loss_calculater(args)
        # self.model.apply(tools.weights_init_normal)

        #------------6. init optimizer, criterion, scheduler, weights-----------------------
        self.optimizer, self.scheduler = build_optimizer(args, len(self.train_dataloader), self.model)
        #------------7. resume training --------------------------------------
        if args.Schedule.resume_path:
            print('Start resume trainning from {}'.format(args.Schedule.resume_path))
            self.__load_model_weights(args.Schedule.resume_path)

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

    def scalar_summary(self, tag, phase, value, step):
        self.writer.add_scalars(tag, {phase: value}, step)

    def train(self):

        print("Train datasets number is : {}".format(len(self.train_dataset)))
        all_iter = (self.epochs - self.start_epoch) * len(self.train_dataloader)
        for epoch in range(self.start_epoch, self.epochs):
            
            self.model.train()
            avg_loss = torch.zeros(3)
            iter_time = 0
            start_time = time.time()

            for i, data in enumerate(self.train_dataloader):
                # torch.cuda.synchronize()
                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                self.optimizer.zero_grad()
                imgs = data['img']
                bboxes = data['annot']
                imgs = imgs.to(self.device)
                bboxes = bboxes.to(self.device)
                # print(i, imgs.shape)
                features = self.model(imgs)
                loss, loss_reg, loss_cls = self.loss_calculater(imgs, features, bboxes)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

                # Update running mean loss
                loss_items = torch.tensor([loss_reg.item(), loss_cls.item(), loss.item()])
                avg_loss = (avg_loss * i + loss_items) / (i + 1)

                print_fre = 10
                # Print batch results
                if i != 0 and i % print_fre == 0:
                    iter_time = iter_time / print_fre
                    eta_seconds = (all_iter - (epoch - self.start_epoch) * len(self.train_dataloader) - (i - 1)) * iter_time
                    eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

                    line = 'Epoch:[{}|{}], Batch:[{}|{}], iter_time:{:.2f}s, loss_avg:{:.2f}, loss_reg:{:.2f}, loss_cls:{:.2f}, lr:{:.2g}'.format(
                        epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, iter_time, avg_loss[2], loss_items[0], loss_items[1], self.optimizer.param_groups[0]['lr'])
                    print(line +', ' + eta_str)
                    iter_time = 0
                    if self.tensorboard:
                        self.scalar_summary("Train_loss/avg_loss", "Train", avg_loss[2], i + epoch * len(self.train_dataloader))
                        self.scalar_summary("Train_loss/lr", "Train", self.optimizer.param_groups[0]['lr'], i+epoch * len(self.train_dataloader))

                end_time = time.time()
                iter_time += end_time - start_time
                start_time = time.time()
                # break
            # self.scheduler.step(avg_loss[2])
            mAP = 0
            if self.save_path:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.__save_model_weights(self.save_path, epoch, mAP)
                print('best mAP : %g' % (self.best_mAP))
            if epoch > 0 and epoch % self.val_intervals == 0:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    if self.dataset == 'coco':
                        aps = coco_eval.evaluate_coco(self.val_dataset, self.model, save_path=self.save_path)
                if self.tensorboard:
                    self.scalar_summary("Train_loss/AP_50", "Train", aps["AP_50"], epoch)


if __name__ == "__main__":

    import sys 
    # sys.argv = ['train.py', '--b', '40', '--device', '0' ]
    default_config_parser = parser = argparse.ArgumentParser(description= 'General Detection config parser')
    parser.add_argument('--config', type=str, default='./config/test.yaml', help="train config file path")
    parser.add_argument('--weight_path', type=str, default='', help='weight file path to pretrain')
    parser.add_argument('--dataset', type=str, default='coco', help='dataset type')
    parser.add_argument('--dataset_path', type=str, default='./dataset/voc2coco', help='path of dataset')
    parser.add_argument('--resume_path', type=str, default='', help='path of model file to resume')
    parser.add_argument('--save_path', type=str, default='', help='save model path')
    parser.add_argument('--pre_train', type=bool, default=True, help='whether to use pre-trained models')
    parser.add_argument('--tensorboard', action='store_true', help='whether to use tensorboard')
    parser.add_argument('--batch_size', '--b', type=int, default=2,  help='mini batch number')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--val_intervals', type=int, default=5,  help='val intervals')
    opt = parser.parse_args()
    load_config(cfg, opt.config)
    # update_opt_to_cfg
    # cfg = parse_args_and_yaml(default_config_parser)
    # cfg = Load_config(opt, opt.config)
    print(cfg)
    Trainer(args = cfg).train()

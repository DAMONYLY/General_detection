# Modification 2022 DAMON
# Copyright 2021-2022 Megvii Inc. 

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from model.utils import get_rank, get_local_rank, get_world_size
from utils.logger import setup_logger
from model.data_load import build_train_dataloader, build_val_dataloader, DataPrefetcher
from model.build_model import build_model
from utils.model_info import get_model_info
from eval.coco_eval import COCO_Evaluater
from utils.optimizer import build_optimizer
from model.loss_calculater import Loss_calculater
from utils.gpu import gpu_mem_usage

class Trainer:
    def __init__(self, args):
        #----------- 1. init ------------------------------------------
        self.start_epoch = 0
        self.best_mAP_info = {'best_mAP': 0.0, 'best_epoch': 0}
        self.epochs = args.Schedule.epochs
        self.save_path = args.Log.save_path
        self.dataset = args.Data.dataset_type
        self.val_intervals = args.Log.val_intervals
        
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank) if torch.cuda.is_available() else "cpu"
        self.tensorboard = args.Log.tensorboard if self.rank == 0 else False
        setup_logger(save_dir=args.Log.save_path, distributed_rank=self.rank)

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'logs'))
        #----------- 2. get dataloader ------------------------------------------
        if self.dataset == 'coco':
            self.train_dataloader = build_train_dataloader(cfg=args.Data.train, 
                                                     batch_size=args.Schedule.device.batch_size,
                                                     num_workers=args.Schedule.device.num_workers,
                                                     is_distributed=self.is_distributed,
                                                     seed=args.Schedule.seed
                                                     )
            self.val_dataloader = build_val_dataloader(cfg=args.Data.test, 
                                                     batch_size=args.Schedule.device.batch_size,
                                                     num_workers=args.Schedule.device.num_workers,
                                                     is_distributed=self.is_distributed
                                                     )
        logger.info("=> Init data prefetcher to speed up dataloader...")
        self.prefetcher = DataPrefetcher(self.train_dataloader, self.device)
        self.max_iter = len(self.train_dataloader)
        #----------- 3. build model -----------------------------------------------
        self.model = build_model(args).to(self.device)
        # logger.info(self.model)
        self.model_info = get_model_info(self.model, args.Data.test.pipeline.input_size)
        logger.info("Model Summary: {}".format(self.model_info))
        #------------4. init optimizer, scheduler-----------------------
        self.optimizer, self.scheduler = build_optimizer(args, len(self.train_dataloader), self.model)
        #------------5. resume training --------------------------------------
        if args.Schedule.resume_path:
            logger.info('Start resume trainning from {}'.format(args.Schedule.resume_path))
            self.load_model_weights(args.Schedule.resume_path)
        #------------6. build loss calculater--------------------------------
        self.loss_calculater = Loss_calculater(args)
        #------------7. build evaluator--------------------------------
        self.evaluator = COCO_Evaluater(self.val_dataloader, self.device, args)
        #------------8. DP mode ------------------------------
        if self.is_distributed:        
            self.model = DDP(self.model, device_ids=[self.local_rank], broadcast_buffers=False)

    def load_model_weights(self, resume_path):
        chkpt = torch.load(resume_path, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mAP_info = chkpt['best_mAP_info']
        del chkpt

    def save_model_weights(self, path, epoch, mAP):
        if mAP['mAP'] > self.best_mAP_info['best_mAP']:
            self.best_mAP_info['best_mAP'] = mAP['mAP']
            self.best_mAP_info['best_epoch'] = mAP['epoch']
        best_weight = os.path.join(path, "best.pt")
        last_weight = os.path.join(path, 'backup_epoch%g.pt'%epoch)
        if self.is_distributed:
            model = self.model.module
        else:
            model = self.model
        chkpt = {'epoch': epoch,
                 'best_mAP_info': self.best_mAP_info,
                 'model': model.state_dict(),
                 'optimizer': self.optimizer.state_dict()
                 }
        torch.save(chkpt, last_weight)
        logger.info('model saved to {}'.format(last_weight))
        if self.best_mAP_info['best_mAP'] == mAP['mAP']:
            torch.save(chkpt['model'], best_weight)
        del chkpt

    def scalar_summary(self, tag, phase, value, step):
        self.writer.add_scalars(tag, {phase: value}, step)


    def train(self):

        logger.info("Train datasets number is : {}".format(len(self.train_dataloader.dataset)))
        all_iter = (self.epochs - self.start_epoch) * self.max_iter
        mAP_info = {'mAP': 0., 'epoch': 0}
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            avg_loss = torch.zeros(3)
            iter_time = 0
            start_time = time.time()

            for i in range(self.max_iter):
                # torch.cuda.synchronize()
                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                self.optimizer.zero_grad()
                data = self.prefetcher.next()
                imgs = data['imgs']
                targets = data['targets']
                # break
                # show_dataset(self.train_dataloader, './test', num = 10)
                features = self.model(imgs)
                loss, loss_reg, loss_cls = self.loss_calculater(features, targets)

                loss.backward()
                self.optimizer.step()
                # Update running mean loss
                loss_items = torch.tensor([loss_reg.item(), loss_cls.item(), loss.item()])
                avg_loss = (avg_loss * i + loss_items) / (i + 1)

                print_fre = 10
                if i != 0 and i % print_fre == 0:
                    iter_time = iter_time / print_fre
                    eta_seconds = (all_iter - (epoch - self.start_epoch) * len(self.train_dataloader) - (i - 1)) * iter_time
                    eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
                    line = 'Epoch:[{}|{}], Batch:[{}|{}], memory_usage:{:.0f}MB, iter_time:{:.2f}s, loss_avg:{:.2f}, loss_reg:{:.2f}, loss_cls:{:.2f}, lr:{:.2g}'.format(
                        epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, gpu_mem_usage(), iter_time, avg_loss[2], loss_items[0], loss_items[1], self.optimizer.param_groups[0]['lr'])
                    logger.info(line + ', ' + eta_str)
                    iter_time = 0
                if self.tensorboard:
                    self.scalar_summary("avg_loss", "Train", avg_loss[2], i + epoch * len(self.train_dataloader))
                    self.scalar_summary("reg_loss", "Train", avg_loss[0], i + epoch * len(self.train_dataloader))
                    self.scalar_summary("cls_loss", "Train", avg_loss[1], i + epoch * len(self.train_dataloader))
                    self.scalar_summary("lr", "Train", self.optimizer.param_groups[0]['lr'], i+epoch * len(self.train_dataloader))

                end_time = time.time()
                iter_time += end_time - start_time
                start_time = time.time()
            
            if epoch > -1 and epoch % self.val_intervals == 0:
                logger.info('*'*20+"Validate"+'*'*20)
                aps = self.evaluator.evalute(self.model, self.save_path)
                if self.rank == 0:
                    mAP_info = {'mAP': aps["AP_50"], 'epoch': epoch}
                if self.tensorboard:
                    self.scalar_summary("AP_50", "Train", aps["AP_50"], epoch)

            if self.save_path and self.rank == 0:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.save_model_weights(self.save_path, epoch, mAP_info)
                logger.info('best mAP:{:.4f} at epoch {:.0f}'.format(self.best_mAP_info['best_mAP'], self.best_mAP_info['best_epoch']))
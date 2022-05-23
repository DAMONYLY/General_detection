from model.build_model import build_model
from utils.gpu import select_device
import torch
from torch.utils.data import DataLoader
import argparse
from utils.tools import *
from utils.model_info import get_model_info
from model.data_load.datasets import CocoDataset
from model.data_load import simple_collater, AspectRatioBasedSampler
from eval.coco_eval import COCO_Evaluater
from utils.config import cfg, load_config
from utils.visualize import *



class Trainer(object):
    def __init__(self, args):
        #----------- 1. get gpu info -----------------------------------------------
        self.device = select_device(args.Schedule.device.gpus)
        self.DP = False
        self.dataset = args.Data.dataset_type

        #----------- 2. get dataset ------------------------------------------
        if self.dataset == 'coco':
            self.dataset = CocoDataset(args.Data.test.dataset_path,
                                    set_name='val2017',
                                    pipeline=args.Data.test.pipeline
                                    )
            sampler = AspectRatioBasedSampler(self.dataset, 
                                                batch_size=args.Schedule.device.batch_size, 
                                                drop_last=False
                                                )
            self.dataloader = DataLoader(self.dataset,
                                            num_workers=args.Schedule.device.num_workers,
                                            batch_sampler=sampler,
                                            collate_fn=simple_collater
                                            )

        #----------- 3. build model -----------------------------------------------
        self.model = build_model(args).to(self.device)
        self.model_info = get_model_info(self.model, args.Data.test.pipeline.input_size)
        print("Model Summary: {}".format(self.model_info))

        #------------4. build evaluator--------------------------------
        self.evaluator = COCO_Evaluater(self.dataloader, self.device, args)

        #------------5. resume training --------------------------------------
        if args.Schedule.resume_path:
            print('=> Loading model checkpoints from {}'.format(args.Schedule.resume_path))
            chkpt = torch.load(args.Schedule.resume_path, map_location=self.device)
            self.model.load_state_dict(chkpt['model'])
            del chkpt
        else:
            raise ValueError('Must have resume_path to load checkpoints to evalute')
        #-------------6. DP mode ------------------------------
        if self.device and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(self.model)
            self.model = model.to(self.device)
            self.DP = True

    def validation(self):    
        aps = self.evaluator.evalute(self.model)
        print(aps)


if __name__ == "__main__":

    import sys 
    # sys.argv = ['train.py', '--b', '40', '--device', '0' ]
    default_config_parser = parser = argparse.ArgumentParser(description= 'General Detection config parser')
    parser.add_argument('--config', type=str, default='./results/Resnet50_lr0.01_atss/experiments.yaml', help="train config file path")
    parser.add_argument('--model_path', type=str, default='./results/Resnet50_lr0.01_atss/backup_epoch29.pt', help="model checkpoints")
    parser.add_argument('--batch_size', type=int, default=1, help="batchsize")
    opt = parser.parse_args()
    load_config(cfg, opt.config, save=False)
    cfg.defrost()
    cfg.Schedule.resume_path = opt.model_path
    cfg.Schedule.device.batch_size = opt.batch_size
    cfg.freeze()
    Trainer(cfg).validation()

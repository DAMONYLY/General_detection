from model.build_model import build_model
import utils.gpu as gpu
import torch
from torch.utils.data import DataLoader
import argparse
from utils.tools import *
from utils import model_info
from model.data_load.datasets import CocoDataset
from model.data_load import simple_collater, AspectRatioBasedSampler
from eval.coco_eval import COCO_Evaluater
from utils.config import cfg, load_config
from utils.visualize import *
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'


class Trainer(object):
    def __init__(self, args):
        #----------- 1. init seed for reproduce -----------------------------------
        init_seeds(args.Schedule.seed)
        #----------- 2. get gpu info -----------------------------------------------
        self.device = gpu.select_device(args.Schedule.device.gpus)
        self.DP = False
        self.dataset = args.Data.dataset_type

        #----------- 3. get dataset ------------------------------------------
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

        #----------- 4. build model -----------------------------------------------
        self.model = build_model(args).to(self.device)
        
        self.model_info = model_info.get_model_info(self.model, args.Data.test.pipeline.input_size)
        print("Model Summary: {}".format(self.model_info))
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        #------------6. build evaluator--------------------------------
        self.evaluator = COCO_Evaluater(self.dataloader, self.device, args)

        #------------8. resume training --------------------------------------
        if args.Schedule.resume_path:
            print('=> Resume trainning from {}'.format(args.Schedule.resume_path))
            self.__load_model_weights(args.Schedule.resume_path)

        #-------------9. DP mode ------------------------------
        if self.device and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(self.model)
            self.model = model.to(self.device)
            self.DP = True


    def __load_model_weights(self, resume_path):
        chkpt = torch.load(resume_path, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
        del chkpt

    def validation(self):    
        aps = self.evaluator.evalute(self.model)
        print(aps)


if __name__ == "__main__":

    import sys 
    # sys.argv = ['train.py', '--b', '40', '--device', '0' ]
    default_config_parser = parser = argparse.ArgumentParser(description= 'General Detection config parser')
    parser.add_argument('--config', type=str, default='./results/test/experiments.yaml', help="train config file path")
    # parser.add_argument('--model_path', type=str, default='./results/Retinanet0.702/backup_epoch49.pt', help="train config file path")
    opt = parser.parse_args()
    load_config(cfg, opt.config, save=False)
    Trainer(args = cfg).validation()

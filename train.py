import argparse
import torch
import random
from loguru import logger
import torch.backends.cudnn as cudnn
from utils.config import cfg, load_config
from model.utils import configure_module, configure_nccl, configure_omp, get_num_devices
from model.core import launch, Trainer


@logger.catch
def main(cfg):
    if cfg.Schedule.seed is not None:
        random.seed(cfg.Schedule.seed)
        torch.manual_seed(cfg.Schedule.seed)
        cudnn.deterministic = True
        logger.warning(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(args=cfg)
    trainer.train()

if __name__ == "__main__":

    import sys 
    # sys.argv = ['train.py', '--b', '40', '--device', '0' ]
    configure_module()
    parser = argparse.ArgumentParser(description= 'General Detection config parser')
    parser.add_argument('--config', type=str, default='./config/retinanet_r50_voc2coco.yaml', help="train config file path")
    parser.add_argument("--dist_url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    opt = parser.parse_args()
    load_config(cfg, opt.config, save=True)
    
    num_gpu = get_num_devices(cfg.Schedule.device.gpus)
    dist_url = "auto" if opt.dist_url is None else opt.dist_url

    launch(
        main,
        num_gpu,
        opt.num_machines,
        opt.machine_rank,
        backend=opt.dist_backend,
        dist_url=dist_url,
        args=cfg,
    )



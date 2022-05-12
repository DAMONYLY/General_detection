'目前是用于构建模型的启动器，根据模型名字选择检测模型'
"2021年11月24日20:31:54"
import torch.nn as nn
from model.detector import General_detector

def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
            
def build_model(cfg):
    if cfg.Model.name == 'General':
        model = General_detector(cfg)
    else:
        raise NotImplementedError
    # model.init_head(1e-2)
    return model
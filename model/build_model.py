'目前是用于构建模型的启动器，根据模型名字选择检测模型'
"2021年11月24日20:31:54"

from model.yolov3 import Yolov3
from model.detector import General_detector
def build(cfg):
    if cfg.MODEL['name'] == 'yolo':
        return Yolov3()
    elif cfg.MODEL['name'] == 'General':
        return General_detector(cfg)
    else:
        raise NotImplementedError
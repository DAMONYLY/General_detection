'目前是用于构建模型的启动器，根据模型名字选择检测模型'
"2021年11月24日20:31:54"
from model.detector import General_detector
            
def build_model(cfg):
    if cfg.Model.name == 'General':
        model = General_detector(cfg)
    else:
        raise NotImplementedError
    return model
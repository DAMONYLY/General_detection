from yacs.config import CfgNode
from contextlib import redirect_stdout
import os 

cfg = CfgNode(new_allowed=True)

cfg.Classes = CfgNode(new_allowed=True)
# NETWORK related params
cfg.Model = CfgNode(new_allowed=True)


# # DATASET related params
cfg.Data = CfgNode(new_allowed=True)

cfg.Schedule = CfgNode(new_allowed=True)

cfg.Log = CfgNode(new_allowed=True)


def load_config(cfg, args_cfg, save=True):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()
    if save and cfg.Log.save_path:
        save_config(cfg, cfg.Log.save_path)

def save_config(cfg, path):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, 'experiments.yaml'), 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

if __name__ == "__main__":
    path = '/opt/General_detection/config/test.yaml'
    load_config(cfg, path)
    print(cfg)
    

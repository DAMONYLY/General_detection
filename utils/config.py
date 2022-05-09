from yacs.config import CfgNode

cfg = CfgNode(new_allowed=True)

# NETWORK related params
cfg.Model = CfgNode(new_allowed=True)


# # DATASET related params
cfg.Data = CfgNode(new_allowed=True)


cfg.Schedule = CfgNode(new_allowed=True)

cfg.Log = CfgNode(new_allowed=True)

cfg.Train = CfgNode(new_allowed=True)
cfg.Test = CfgNode(new_allowed=True)


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()



if __name__ == "__main__":
    path = '/opt/General_detection/config/test.yaml'
    load_config(cfg, path)
    print(cfg)
    

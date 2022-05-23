from yacs.config import CfgNode
import os 
import yaml
from loguru import logger

cfg = CfgNode(new_allowed=True)

cfg.Classes = CfgNode(new_allowed=True)
# NETWORK related params
cfg.Model = CfgNode(new_allowed=True)


# # DATASET related params
cfg.Data = CfgNode(new_allowed=True)

cfg.Schedule = CfgNode(new_allowed=True)

cfg.Log = CfgNode(new_allowed=True)

_VALID_TYPES = {tuple, list, str, int, float, bool}

def dump_to_yaml(cfg, **kwargs):
    """Dump to a string."""

    def convert_to_dict(cfg_node, key_list=[]):
        """ Convert a config node to dictionary """
        if not isinstance(cfg_node, CfgNode):
            if type(cfg_node) not in _VALID_TYPES:
                logger.warning("Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES), )
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = convert_to_dict(v, key_list + [k])
            return cfg_dict

    self_as_dict = convert_to_dict(cfg, [])
    return yaml.safe_dump(self_as_dict, **kwargs)


def load_config(cfg, args_cfg, save=True):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()
    if save and cfg.Log.save_path:
        save_config(cfg, cfg.Log.save_path)


def save_config(cfg, path):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, "experiments.yaml"), "w") as f:
        dump_to_yaml(cfg, stream=f)



if __name__ == "__main__":
    path = '/opt/General_detection/config/test.yaml'
    load_config(cfg, path)
    print(cfg)
    

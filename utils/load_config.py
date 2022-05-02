import yaml

'''
2022年4月27日15:21:02
copy from https://blog.51cto.com/u_15127596/4233240
将config送入args合并
'''
# default_config_parser = parser = argparse.ArgumentParser(
#     description='Training Config', add_help=False)
# parser.add_argument(
#     '-c',
#     '--config_yaml',
#     default=
#     './config/test.yaml',
#     type=str,
#     metavar='FILE',
#     help='YAML config file specifying default arguments')

# YAML should override the argparser's content
def parse_args_and_yaml(given_parser=None):
    if given_parser == None:
        raise ValueError('Please give the ArgumentParser')
    given_configs, remaining = given_parser.parse_known_args()
    if given_configs.config:
        with open(given_configs.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            a = Config(cfg) # 方法1
            # a = wrap(cfg) # 方法2
            given_parser.set_defaults(**a)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = given_parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    # args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # return args, args_text
    return args


'''
方法1: https://gist.github.com/mmerickel/ff4c6faf867d72c1f19c
'''  

# 将args送入config合并
def Load_config(cfg, path):
    with open(path,'r') as f:
        config=yaml.load(f)
        a = Config(config)
        # a = Dictate(config)
    for arg in vars(cfg):
        value = getattr(cfg, arg)
        a.add(arg, value)
    print("getting hypeparameters:\n",a, cfg)
    return a

 
class Config(dict):
    def __init__(self,config):
        super(Config, self).__init__(config)
        for key in self:
            item = self[key]
            if isinstance(item, list):
                for idx, it in enumerate(item):
                    if isinstance(it, dict):
                        item[idx] = Config(it)
            elif isinstance(item, dict):
                self[key] = Config(item)

    def __getattr__(self, key):
        return self[key]

    def get(self,name,default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def has(self,name):
        try:
            return hasattr(self,name)
        except KeyError:
            return False

    def add(self,name=None,value=None):
        setattr(self, name, value)
        # if not hasattr(self,name):
            
        #     with open(self.configFile,'a') as f:
        #         f.write(str(name)+": "+str(value)+'\n')
        # else:
        #     print('\'{}\' already exists in \'config\' , its values is {} , maybe you just want to change its value?'.format(name,getattr(self,name)))



'''
方法2: https://gist.github.com/mmerickel/ff4c6faf867d72c1f19c
'''    
import collections

def wrap(value):
    """
    The top-level API for wrapping an arbitrary object.
    
    This only works for ``dict``, ``list`` and ``tuple`` types. If you want
    to wrap other types you may write your own ``wrap`` and pass ``wrapper=``
    to ``DictProxy`` and ``ListProxy``.
    
    """
    if isinstance(value, dict):
        return DictProxy(value)
    if isinstance(value, (tuple, list)):
        return ListProxy(value)
    return value


class DictProxy(collections.Mapping):
    """
    A proxy for a dictionary that allows attribute access to underlying keys.
    
    You may pass a custom ``wrapper`` to override the logic for wrapping
    various custom types.
    
    """
    def __init__(self, obj, wrapper=wrap):
        self.obj = obj
        self.wrapper = wrapper
        
    def __getitem__(self, key):
        return self.wrapper(self.obj[key])
        
    def __len__(self):
        return self.obj.__len__()
        
    def __iter__(self):
        return self.obj.__iter__()
        
    def __getattr__(self, key):
        try:
            return self.wrapper(getattr(self.obj, key))
        except AttributeError:
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)
            
class ListProxy(collections.Sequence):
    """
    A proxy for a list that allows for wrapping items.
    
    You may pass a custom ``wrapper`` to override the logic for wrapping
    various custom types.
    
    """
    def __init__(self, obj, wrapper=wrap):
        self.obj = obj
        self.wrapper = wrapper
        
    def __getitem__(self, key):
        return self.wrapper(self.obj[key])
        
    def __len__(self):
        return self.obj.__len__()
        

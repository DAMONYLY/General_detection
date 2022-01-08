import yaml

def Load_config(cfg, path):
    with open(path,'r') as f:
        config=yaml.load(f)
        a = Config(config)
        print("getting hypeparameters:\n",a)
        # return a
    for arg in vars(cfg):
        value = getattr(cfg, arg)
        a.add(arg, value)
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

    def has(self,name):
        return hasattr(self,name)

    def add(self,name=None,value=None):
        setattr(self, name, value)
        # if not hasattr(self,name):
            
        #     with open(self.configFile,'a') as f:
        #         f.write(str(name)+": "+str(value)+'\n')
        # else:
        #     print('\'{}\' already exists in \'config\' , its values is {} , maybe you just want to change its value?'.format(name,getattr(self,name)))
        
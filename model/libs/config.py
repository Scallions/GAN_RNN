import yaml

from . import manager

class Config:
    def __init__(self, path):
        self._parse_yaml(path)


    def _parse_yaml(self, path):
        with open(path, 'r') as f:
            dic = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in dic.items():
            if self._is_meta_type(v):
                dic[k] = self._load_object(v)
        self.__dict__.update(dic)

    def _is_meta_type(self,v):
        return isinstance(v, dict) and 'type' in v.keys()

    def _load_object(self, cfg):
        cfg = cfg.copy()
        component = self._load_component(cfg.pop('type'))
        params = {}
        for k, v in cfg.items():
            if self._is_meta_type(v):
                params[k] = self._load_object(v)
            elif isinstance(v, list):
                params[k] = [
                    self._load_object(item)
                    if self._is_meta_type(item) else item
                    for item in v
                ]
            else:
                params[k] = v

        return component(**params)

    def _load_component(self, name):
        com_managers = [
            manager.MODELS,
            manager.DATASETS,
        ]
        for mgr in com_managers:
            if name in mgr.components_dict.keys():
                return mgr.components_dict[name]
        else:
            raise RuntimeError(f'{name} not found')

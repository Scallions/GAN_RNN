

class ComponentManager:
    def __init__(self, name=None):
        self._name = name
        self._components_dict = dict()


    def add_component(self, component):
        self._components_dict[component.__name__] = component
        return component

    def __getitem__(self, key):
        if key not in self._components_dict.keys():
            raise KeyError(f'{key} not found in {self._name}')
        return self._components_dict[key]

    @property
    def components_dict(self):
        return self._components_dict

    @property
    def name(self):
        return self._name


MODELS = ComponentManager('models')
TRAINERS = ComponentManager('trainer')
EXPS = ComponentManager('experiments')
DATASETS = ComponentManager('datasets')
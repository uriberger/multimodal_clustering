import torch
import abc
import os
from loggable_object import LoggableObject


class ModelWrapper(LoggableObject):

    def __init__(self, device, config, model_dir, indent, name):
        """ The init function is used both for creating a new instance (when config is specified), and for loading
        saved instances (when config is None). """
        super(ModelWrapper, self).__init__(indent)
        self.device = device
        self.model_dir = model_dir
        self.dump_path = os.path.join(self.model_dir, name)

        need_to_load = (config is None)
        if need_to_load:
            self.load_config()
        else:
            self.config = config
        self.model = self.generate_model()
        if need_to_load:
            self.load_model()

    @abc.abstractmethod
    def generate_model(self):
        return

    @abc.abstractmethod
    def dump_model(self):
        return

    @abc.abstractmethod
    def load_model(self):
        return

    def get_dump_path(self):
        return self.dump_path

    def get_config_path(self):
        return self.get_dump_path() + '.cfg'

    def get_model_path(self):
        return self.get_dump_path() + '.mdl'

    def dump_config(self):
        torch.save(self.config, self.get_config_path())

    def load_config(self):
        config_path = self.get_config_path()
        if not os.path.isfile(config_path):
            self.log_print('Model was instantiated without configuration, and no config file was found. Stopping!')
            self.log_print('The config path was ' + config_path)
            assert False
        self.config = torch.load(config_path)

    def dump(self):
        self.dump_config()
        self.dump_model()
        return

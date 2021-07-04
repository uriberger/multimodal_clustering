import torch
import abc
import os
from loggable_object import LoggableObject


class ModelWrapper(LoggableObject):

    def __init__(self, device, config, model_dir, indent, name):
        super(ModelWrapper, self).__init__(indent)
        self.config = config
        self.device = device
        self.model_dir = model_dir
        self.dump_path = os.path.join(self.model_dir, name)
        self.cached_output = None

    def load_model_if_needed(self):
        if os.path.exists(self.get_model_path()):
            # Load from dumped file
            loaded_config = self.load_config()

            # Make sure the given configuration matches the loaded one
            if loaded_config != self.config:
                self.log_print('Provided config differs from loaded config!')
                assert False

            self.load_model()

    @abc.abstractmethod
    def training_step(self, inputs, labels):
        return

    @abc.abstractmethod
    def inference(self, inputs):
        return

    @abc.abstractmethod
    def dump(self):
        return

    @abc.abstractmethod
    def load_model(self, model_dir):
        return

    @abc.abstractmethod
    def predict_concept_indicators(self):
        return

    def predict_concept_lists(self):
        concept_indicators = self.predict_concept_indicators()
        predicted_concept_lists = [[x for x in range(self.config.concept_num) if concept_indicators[i, x] == 1]
                                   for i in range(concept_indicators.shape[0])]

        return predicted_concept_lists

    def get_dump_path(self):
        return self.dump_path

    def get_config_path(self):
        return self.get_dump_path() + '.cfg'

    def get_model_path(self):
        return self.get_dump_path() + '.mdl'

    def dump_config(self):
        torch.save(self.config, self.get_config_path())

    def load_config(self):
        return torch.load(self.get_config_path())

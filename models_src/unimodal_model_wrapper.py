import abc
from models_src.model_wrapper import ModelWrapper
import torch.nn as nn


class UnimodalModelWrapper(ModelWrapper):

    def __init__(self, device, config, model_dir, indent, name):
        """ The init function is used both for creating a new instance (when config is specified), and for loading
        saved instances (when config is None). """
        super(UnimodalModelWrapper, self).__init__(device, config, model_dir, indent, name)
        self.cached_output = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.cached_loss = None

    @abc.abstractmethod
    def generate_model(self):
        return

    @abc.abstractmethod
    def training_step(self, inputs, labels):
        return

    @abc.abstractmethod
    def inference(self, inputs):
        return

    @abc.abstractmethod
    def dump_model(self):
        return

    @abc.abstractmethod
    def load_model(self):
        return

    @abc.abstractmethod
    def predict_concept_indicators(self):
        return

    def predict_concept_lists(self):
        concept_indicators = self.predict_concept_indicators()
        predicted_concept_lists = [[x for x in range(self.config.concept_num) if concept_indicators[i, x] == 1]
                                   for i in range(concept_indicators.shape[0])]

        return predicted_concept_lists

    def print_info_on_loss(self):
        return 'Loss: ' + str(self.cached_loss)

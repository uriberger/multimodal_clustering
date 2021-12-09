import abc
from models_src.model_wrapper import ModelWrapper
import torch.nn as nn


class UnimodalModelWrapper(ModelWrapper):

    def __init__(self, device, config, model_dir, model_name, indent):
        """ The init function is used both for creating a new instance (when config is specified), and for loading
        saved instances (when config is None). """
        super(UnimodalModelWrapper, self).__init__(device, config, model_dir, model_name, indent)
        self.cached_output = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.cached_loss = None

    @abc.abstractmethod
    def generate_underlying_model(self):
        return

    @abc.abstractmethod
    def training_step(self, inputs, labels):
        return

    @abc.abstractmethod
    def inference(self, inputs):
        return

    @abc.abstractmethod
    def dump_underlying_model(self):
        return

    @abc.abstractmethod
    def load_underlying_model(self):
        return

    @abc.abstractmethod
    def predict_cluster_indicators(self):
        return

    def predict_cluster_lists(self):
        cluster_indicators = self.predict_cluster_indicators()
        predicted_cluster_lists = [[x for x in range(self.config.cluster_num) if cluster_indicators[i, x] == 1]
                                   for i in range(cluster_indicators.shape[0])]

        return predicted_cluster_lists

    def print_info_on_loss(self):
        return 'Loss: ' + str(self.cached_loss)

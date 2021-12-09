###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from models_src.model_wrapper import ModelWrapper
import torch.nn as nn


class BimodalModelWrapper(ModelWrapper):
    """ This is the base class for bimodal model wrapper.
        The visual wrapper and text wrapper will inherit from this class. """

    def __init__(self, device, config, model_dir, model_name, indent):
        super(BimodalModelWrapper, self).__init__(device, config, model_dir, model_name, indent)
        self.cached_output = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.cached_loss = None

    # Abstract methods

    """Perform a single training step given the provided inputs and labels. """

    @abc.abstractmethod
    def training_step(self, inputs, labels):
        return

    """Run inference given the provided inputs. """

    @abc.abstractmethod
    def inference(self, inputs):
        return

    """Predict a list of N bits (where N is the total number of clusters), where the ith entry is 1 iff the input to
    the last inference is associated with the ith cluster. """

    @abc.abstractmethod
    def predict_cluster_indicators(self):
        return

    # Implemented methods

    """Same as predict_cluster_indicators but here its in the form of the list of cluster indices. """

    def predict_cluster_lists(self):
        cluster_indicators = self.predict_cluster_indicators()
        predicted_cluster_lists = [[x for x in range(self.config.cluster_num) if cluster_indicators[i, x] == 1]
                                   for i in range(cluster_indicators.shape[0])]

        return predicted_cluster_lists

    def print_info_on_loss(self):
        return 'Loss: ' + str(self.cached_loss)

    # Abstract methods inherited from ModelWrapper class

    @abc.abstractmethod
    def generate_underlying_model(self):
        return

    @abc.abstractmethod
    def dump_underlying_model(self):
        return

    @abc.abstractmethod
    def load_underlying_model(self):
        return

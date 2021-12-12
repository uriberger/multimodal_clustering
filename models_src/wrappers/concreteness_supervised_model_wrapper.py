###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
from models_src.wrappers.model_wrapper import ModelWrapper
from models_src.underlying_models.concreteness_supervised_model import ConcretenessSupervisedModel
from dataset_builders.concreteness_dataset import generate_concreteness_dataset


class ConcretenessSupervisedWrapper(ModelWrapper):
    """ This class wraps the ConcretenessSupervisedModel class. """

    def __init__(self, device, config, model_dir, model_name, indent):
        super(ConcretenessSupervisedWrapper, self).__init__(device, config, model_dir, model_name, indent)

    # Abstract methods inherited from ModelWrapper class

    def generate_underlying_model(self):
        return ConcretenessSupervisedModel(
            self.config.use_pos_tags,
            self.config.use_suffix,
            self.config.use_embeddings,
            self.config.suffix_num,
            self.config.fast_text_dim,
            self.indent
        )

    def dump_underlying_model(self):
        torch.save(self.underlying_model, self.get_underlying_model_path())

    def load_underlying_model(self):
        self.underlying_model = torch.load(self.get_underlying_model_path())

    # Current class specific functionality

    def train_model(self):
        training_set = generate_concreteness_dataset()
        self.word_to_conc_pred = self.underlying_model.train(training_set)

    def estimate_word_concreteness(self, sentences):
        res = []
        for sentence in sentences:
            res.append([])
            for token in sentence:
                if token in self.word_to_conc_pred:
                    # Never seen this token before
                    res[-1].append(0)
                else:
                    res[-1].append(self.word_to_conc_pred[token])

        return res

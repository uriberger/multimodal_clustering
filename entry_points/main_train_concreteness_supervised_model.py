###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

# General
import os
import torch
from utils.general_utils import log_print, init_entry_point, project_root_dir, default_model_name

# Dataset
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_configs.concreteness_supervised_model_config import ConcretenessSupervisedModelConfig
from models_src.wrappers.concreteness_supervised_model_wrapper import ConcretenessSupervisedModelWrapper


""" The entry point for training a concreteness prediction supervised model, according the paper
"Predicting  word  concreteness  and  imagery" by Jean Charbonnier and Christian Wartena.
"""


def main_train_concreteness_supervised_model(write_to_log):
    function_name = 'main_train_concreteness_supervised_model'
    timestamp = init_entry_point(write_to_log)

    model_config = ConcretenessSupervisedModelConfig()
    # model_config = ConcretenessSupervisedModelConfig(use_embeddings=False)
    log_print(function_name, 0, str(model_config))

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)
    training_set_config = DatasetConfig(1)
    training_set, _, _ = dataset_builder.build_dataset(training_set_config)
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Training model...')
    model_root_dir = os.path.join(project_root_dir, timestamp)
    model = ConcretenessSupervisedModelWrapper(torch.device('cpu'), model_config, model_root_dir, default_model_name, 1)
    model.train_model()
    model.dump()
    log_print(function_name, 0, 'Finished training model')

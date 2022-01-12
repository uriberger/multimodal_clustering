###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
import torch
import gensim.downloader as api
from utils.general_utils import log_print, models_dir, text_dir, init_entry_point

# Dataset
from dataset_builders.category_dataset import CategoryDatasetBuilder
from dataset_builders.swow_dataset import SWOWDatasetBuilder
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig

# Metric
from metrics.word_association_metric import WordAssociationMetric

# Models
from models_src.wrappers.text_model_wrapper import \
    TextCountsModelWrapper, \
    TextOnlyCountsModelWrapper, \
    TextRandomModelWrapper
from models_src.wrappers.word_embedding_clustering_model_wrapper import \
    W2VClusteringWrapper, \
    BERTClusteringWrapper, \
    CLIPClusteringWrapper
from models_src.model_configs.cluster_model_config import ClusterModelConfig


""" This entry point evaluated association strength between pairs of words in a clustering solution. """


def main_association_evaluation(write_to_log, model_type, model_name):
    function_name = 'main_association_evaluation'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')

    category_dataset = CategoryDatasetBuilder(1).build_dataset()
    # Filter the category dataset to contain only words with which all of the evaluated models are familiar
    all_words = [x for outer in category_dataset.values() for x in outer]

    # 1. Only words that occurred in the COCO training set
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)
    training_set_config = DatasetConfig(1)
    training_set, _, _ = dataset_builder.build_dataset(training_set_config)
    token_count = training_set.get_token_count()
    all_words = [x for x in all_words if x in token_count]

    # 2. Only words with which word2vec is familiar
    w2v_model = api.load("word2vec-google-news-300")
    all_words = [x for x in all_words if x in w2v_model.key_to_index]

    # Filter the dataset
    word_dict = {x: True for x in all_words}
    category_dataset = {x[0]: [y for y in x[1] if y in word_dict] for x in category_dataset.items()}

    swow_dataset = SWOWDatasetBuilder(all_words, 1).build_dataset()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Testing...')

    text_model_dir = os.path.join(models_dir, text_dir)
    if model_type == 'multimodal_clustering':
        model = TextCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    elif model_type == 'text_only':
        model_name = 'text_only_baseline'
        model = TextOnlyCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    elif model_type == 'random':
        model = TextRandomModelWrapper(torch.device('cpu'), ClusterModelConfig(cluster_num=41), 1)
    elif model_type == 'w2v':
        model = W2VClusteringWrapper(torch.device('cpu'), all_words)
    elif model_type == 'bert':
        model = BERTClusteringWrapper(torch.device('cpu'), all_words)
    elif model_type == 'clip':
        model = CLIPClusteringWrapper(torch.device('cpu'), all_words)

    metric = WordAssociationMetric(model, all_words, swow_dataset)
    log_print(function_name, 1, metric.report())

    log_print(function_name, 0, 'Finished testing')

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
from utils.general_utils import log_print, init_entry_point, project_root_dir

# Dataset
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_configs.cluster_model_config import ClusterModelConfig

# Executors
from executors.trainers.multimodal_clustering_model_trainer import MultimodalClusteringModelTrainer


""" The entry point for filtering unwanted images from the MSCOCO dataset.
"""


def main_filter_unwanted_coco_images(write_to_log=False):
    function_name = 'main_filter_unwanted_coco_images'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Filtering...')
    dataset_builder.filter_unwanted_images()
    log_print(function_name, 0, 'Finished filtering')

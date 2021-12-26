import os
import torch
from utils.general_utils import log_print, models_dir, text_dir, init_entry_point

# Dataset
from dataset_builders.category_dataset import CategoryDatasetBuilder

# Metric
from metrics.categorization_metric import CategorizationMetric

# Models
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper, TextOnlyCountsModelWrapper


def main_categorization_evaluation(write_to_log):
    function_name = 'main_categorization_evaluation'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    category_dataset = CategoryDatasetBuilder(1).build_dataset()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Testing...')

    # model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'
    model_name = 'text_only_baseline'

    text_model_dir = os.path.join(models_dir, text_dir)
    # model = TextCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    model = TextOnlyCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    metric = CategorizationMetric(model, category_dataset, ignore_unknown_words=False)
    log_print(function_name, 1, metric.report())

    log_print(function_name, 0, 'Finished testing')

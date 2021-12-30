###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
from utils.general_utils import log_print, init_entry_point

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Executors
from executors.evaluators.visual_evaluators.random_heatmap_evaluator import RandomHeatmapEvaluator

""" This entry point evaluates the heatmap metric using random guesses. """


def main_random_heatmap_evaluation(write_to_log):
    function_name = 'main_random_heatmap_evaluation'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, slice_str, _ = create_dataset_builder(dataset_name)

    test_set_config = DatasetConfig(1, slice_str=slice_str, include_gt_classes=True, include_gt_bboxes=True)
    test_set, gt_classes_file_path, gt_bboxes_file_path = dataset_builder.build_dataset(test_set_config)
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Testing...')

    evaluator = RandomHeatmapEvaluator(test_set, gt_classes_file_path, gt_bboxes_file_path, 1)
    evaluator.evaluate()
    log_print(function_name, 0, 'Finished testing')

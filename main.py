###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import argparse
import os

from entry_points.main_tune_joint_model_parameters import main_tune_joint_model_parameters
from entry_points.main_train_joint_model import main_train_joint_model
from entry_points.main_filter_unwanted_coco_images import main_filter_unwanted_coco_images
from entry_points.main_categorization_evaluation import main_categorization_evaluation
from entry_points.main_concreteness_evaluation import main_concreteness_evaluation
from entry_points.main_train_text_only_baseline import main_train_text_only_baseline
from entry_points.main_train_concreteness_supervised_model import main_train_concreteness_supervised_model
from entry_points.main_visual_evaluation import main_visual_evaluation
from entry_points.main_text_evaluation import main_text_evaluation
from entry_points.main_random_heatmap_evaluation import main_random_heatmap_evaluation
from entry_points.main_clip_classification_evaluation import main_clip_classification_evaluation
from entry_points.main_find_clip_positive_threshold import main_find_clip_positive_threshold

from dataset_builders.dataset_builder import DatasetBuilder

""" Main entry point. """

parser = argparse.ArgumentParser(description='Train and evaluate a multimodal word learning model.')
parser.add_argument('--utility', type=str, default='train_joint_model', dest='utility',
                    help='the wanted utility')
parser.add_argument('--write_to_log', action='store_true', default=False, dest='write_to_log',
                    help='redirect output to a log file')
parser.add_argument('--datasets_dir', type=str, default=os.path.join('..', 'datasets'), dest='datasets_dir',
                    help='the path to the datasets dir')

args = parser.parse_args()
utility = args.utility
write_to_log = args.write_to_log
datasets_dir = args.datasets_dir
DatasetBuilder.set_datasets_dir(datasets_dir)

if utility == 'tune_joint_model_parameters':
    main_tune_joint_model_parameters(write_to_log)
elif utility == 'train_joint_model':
    main_train_joint_model(write_to_log)
elif utility == 'filter_unwanted_coco_images':
    main_filter_unwanted_coco_images(write_to_log)
elif utility == 'categorization_evaluation':
    main_categorization_evaluation(write_to_log)
elif utility == 'concreteness_evaluation':
    main_concreteness_evaluation(write_to_log)
elif utility == 'train_text_only_baseline':
    main_train_text_only_baseline(write_to_log)
elif utility == 'train_concreteness_supervised_model':
    main_train_concreteness_supervised_model(write_to_log)
elif utility == 'visual_evaluation':
    main_visual_evaluation(write_to_log)
elif utility == 'text_evaluation':
    main_text_evaluation(write_to_log)
elif utility == 'random_heatmap_evaluation':
    main_random_heatmap_evaluation(write_to_log)
elif utility == 'clip_classification_evaluation':
    main_clip_classification_evaluation(write_to_log)
elif utility == 'find_clip_positive_threshold':
    main_find_clip_positive_threshold(write_to_log)
else:
    print('Unknown utility: ' + str(utility))
    print('Please choose one of: bbox_demonstration, categorization_evaluation, concreteness_evaluation, ' +
          'heatmap_demonstration, heatmap_evaluation, train_joint_model, tune_joint_model_parameters, ' +
          'two_phase_bbox, filter_unwanted_coco_images, train_text_only_baseline, visual_evaluation, ' +
          'text_evaluation, train_concreteness_supervised_model, random_heatmap_evaluation, ' +
          'clip_classification_evaluation, find_clip_positive_threshold')

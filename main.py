###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import argparse

from entry_points.main_tune_joint_model_parameters import main_tune_joint_model_parameters

""" Main entry point. """

parser = argparse.ArgumentParser(description='Train and evaluate a multimodal word learning model.')
parser.add_argument('--utility', type=str, default='train_joint_model', dest='utility',
                    help='the wanted utility')
parser.add_argument('--write_to_log', action='store_true', default=False, dest='write_to_log',
                    help='redirect output to a log file')

args = parser.parse_args()
utility = args.utility
write_to_log = args.write_to_log

if utility == 'tune_joint_model_parameters':
    main_tune_joint_model_parameters(write_to_log)
else:
    print('Unknown utility: ' + str(utility))
    print('Please choose one of: bbox_demonstration, categorization_evaluation, concreteness_evaluation, ' +
          'heatmap_demonstration, heatmap_evaluation, train_join_model, tune_joint_model_parameters, ' +
          'two_phase_bbox, visual_model')

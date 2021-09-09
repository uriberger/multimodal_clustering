import os
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Executors
from executors.visual_evaluators.clustering_evaluators.evaluate_clustering_single_label import ClusteringSingleLabelEvaluator
from executors.visual_evaluators.clustering_evaluators.evaluate_clustering_multi_label import ClusteringMultiLabelEvaluator
from executors.visual_evaluators.prompt_evaluators.evaluate_prompt_single_label import PromptSingleLabelEvaluator
from executors.visual_evaluators.prompt_evaluators.evaluate_prompt_multi_label import PromptMultiLabelEvaluator
from executors.visual_evaluators.evaluate_concepts_visual_model import VisualConceptEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_visual_model'
os.mkdir(timestamp)
set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'ImageNet'
dataset_creator, slice_str, multi_label = create_dataset_builder(dataset_name)

test_set_config = DatasetConfig(1, slice_str=slice_str, include_gt_classes=True)
if multi_label:
    test_set, gt_classes_file, _ = dataset_creator.build_dataset(test_set_config)
else:
    test_set = dataset_creator.build_dataset(test_set_config)
class_mapping = dataset_creator.get_class_mapping()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

# model_type = 'pretrained'
# model_str = 'resnet18'
# model_type = 'clip'
# model_str = 'RN50'
# model_str = 'ViT-B/32'
model_type = 'unimodal'
model_str = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'
# model_type = 'simclr'
# model_str = 'simclr_resnet_15_epochs'

# evaluate_method = 'clustering'
evaluate_method = 'prompt'
# evaluate_method = 'concepts'
if evaluate_method == 'clustering':
    if multi_label:
        evaluator = ClusteringMultiLabelEvaluator(test_set, class_mapping, gt_classes_file, model_type, model_str, 1)
    else:
        evaluator = ClusteringSingleLabelEvaluator(test_set, class_mapping, model_type, model_str, 1)
elif evaluate_method == 'prompt':
    if multi_label:
        evaluator = PromptMultiLabelEvaluator(test_set, class_mapping, gt_classes_file, model_type, model_str, 1)
    else:
        evaluator = PromptSingleLabelEvaluator(test_set, class_mapping, model_type, model_str, 1)
elif evaluate_method == 'concepts':
    if not multi_label:
        gt_classes_file = None
    evaluator = VisualConceptEvaluator(test_set, class_mapping, gt_classes_file, model_str, 1)

evaluator.evaluate()
log_print(function_name, 0, 'Finished testing')

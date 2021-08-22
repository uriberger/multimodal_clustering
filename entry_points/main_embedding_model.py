import os
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from dataset_builders.cifar import Cifar10, Cifar100
from dataset_builders.imagenet import ImageNet
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.embedding_evaluators.clustering_evaluators.evaluate_clustering_single_label import ClusteringSingleLabelEvaluator
from executors.embedding_evaluators.clustering_evaluators.evaluate_clustering_multi_label import ClusteringMultiLabelEvaluator
from executors.embedding_evaluators.prompt_evaluators.evaluate_prompt_single_label import PromptSingleLabelEvaluator
from executors.embedding_evaluators.prompt_evaluators.evaluate_prompt_multi_label import PromptMultiLabelEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_embedding_model'
os.mkdir(timestamp)
set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'COCO'
root_dir = os.path.join('..', 'datasets', dataset_name)
if dataset_name == 'cifar-10':
    dataset_generator = Cifar10(root_dir, 1)
    slice_str = 'test'
    multi_label = False
elif dataset_name == 'cifar100':
    dataset_generator = Cifar100(root_dir, 1)
    slice_str = 'test'
    multi_label = False
elif dataset_name == 'ImageNet':
    dataset_generator = ImageNet(root_dir, 1)
    slice_str = 'val'
    multi_label = False
elif dataset_name == 'COCO':
    dataset_generator = Coco(root_dir, 1)
    slice_str = 'val'
    multi_label = True
else:
    assert False

test_set_config = DatasetConfig(1, slice_str=slice_str, include_gt_classes=True)
if multi_label:
    test_set, gt_classes_file, _ = dataset_generator.build_dataset(test_set_config)
else:
    test_set = dataset_generator.build_dataset(test_set_config)
class_mapping = dataset_generator.get_class_mapping()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

# model_type = 'pretrained'
# model_str = 'resnet18'
# model_type = 'clip'
# model_str = 'RN50'
# model_str = 'ViT-B/32'
# model_type = 'unimodal'
# model_str = 'resnet_non_pretrained_noun_th_0.03_conc_num_100'
model_type = 'simclr'
model_str = 'simclr_resnet_15_epochs'

evaluate_method = 'clustering'
# evaluate_method = 'prompt'
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

evaluator.evaluate()
log_print(function_name, 0, 'Finished testing')

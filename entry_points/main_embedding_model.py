import os
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from dataset_builders.cifar import Cifar10, Cifar100
from dataset_builders.imagenet import ImageNet
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.embedding_evaluators.evaluate_clustering import ClusteringEvaluator
from executors.embedding_evaluators.evaluate_prompt_single_label import PromptSingleLabelEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_embedding_model'
os.mkdir(timestamp)
set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
# cifar10_dir = os.path.join('..', 'datasets', 'cifar-10')
# dataset_generator = Cifar10(cifar10_dir, 1)
# cifar100_dir = os.path.join('..', 'datasets', 'cifar100')
# dataset_generator = Cifar100(cifar100_dir, 1)
imagenet_dir = os.path.join('..', 'datasets', 'ImageNet')
dataset_generator = ImageNet(imagenet_dir, 1)

test_set_config = DatasetConfig(1, slice_str='val')
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
    evaluator = ClusteringEvaluator(test_set, class_mapping, model_type, model_str, 1)
elif evaluate_method == 'prompt':
    evaluator = PromptSingleLabelEvaluator(test_set, class_mapping, model_type, model_str, 1)

evaluator.evaluate()
log_print(function_name, 0, 'Finished testing')

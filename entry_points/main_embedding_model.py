import os
import torch
import torch.nn as nn
import torchvision.models as models
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from dataset_builders.cifar10 import Cifar10, Cifar100
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.evaluate_embedding_model import EmbeddingModelEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_evaluate_dumped_inference'
os.mkdir(timestamp)
# set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
cifar10_dir = os.path.join('..', 'datasets', 'cifar-10')
cifar10 = Cifar10(cifar10_dir, 1)
# cifar100_dir = os.path.join('..', 'datasets', 'cifar100')
# cifar100 = Cifar100(cifar100_dir, 1)

test_set_config = DatasetConfig(1, slice_str='test', normalize_images=False)
test_set = cifar10.build_dataset(test_set_config)
class_mapping = cifar10.get_class_mapping()
# test_set = cifar100.build_dataset(test_set_config)
# class_mapping = cifar100.get_class_mapping()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

# Load our external model
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(in_features=512, out_features=56402)
# model.load_state_dict(torch.load('image_embedder_3_epochs'))
# model = models.resnet18(pretrained=True)
# model.fc = nn.Identity()
# model.eval()

# evaluator = EmbeddingModelEvaluator(test_set, class_mapping, 'pretrained', 'resnet18', 1)
# evaluator = EmbeddingModelEvaluator(test_set, class_mapping, 'clip', 'RN50', 1)
evaluator = EmbeddingModelEvaluator(test_set, class_mapping, 'unimodal', 'resnet_non_pretrained_noun_th_0.03_conc_num_100', 1)
evaluator.evaluate()
log_print(function_name, 0, 'Finished testing')

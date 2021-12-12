import os
import torch
from utils.general_utils import log_print, models_dir, text_dir, init_entry_point, project_root_dir

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder
from dataset_builders.category_dataset import generate_fountain_category_dataset

# Metric
from metrics import CategorizationMetric

# Models
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper

function_name = 'main_categorization_evaluation'
timestamp = init_entry_point(True)

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'COCO'
coco_builder, slice_str, multi_label = create_dataset_builder(dataset_name)

training_set_config = DatasetConfig(1)
training_set, _, _ = coco_builder.build_dataset(training_set_config)
token_count = training_set.get_token_count()
category_dataset = generate_fountain_category_dataset()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

# model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'
model_name = 'model'

text_model_dir = os.path.join(project_root, models_dir, text_dir)
model = TextCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
# model = TextualSimpleCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
metric = CategorizationMetric(model, category_dataset, ignore_unknown_words=True)
log_print(function_name, 1, metric.report())

log_print(function_name, 0, 'Finished testing')

import os
import torch
from utils.general_utils import log_print, set_write_to_log, models_dir, text_dir
from datetime import datetime

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder
from dataset_builders.concreteness_dataset import generate_concreteness_dataset

# Metric
from metrics import ConcretenessPredictionMetric

# Models
from models_src.textual_model_wrapper import TextualCountsModelWrapper


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_concreteness evaluation'
os.mkdir(timestamp)
# set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'COCO'
coco_builder, slice_str, multi_label = create_dataset_builder(dataset_name)

training_set_config = DatasetConfig(1)
training_set, _, _ = coco_builder.build_dataset(training_set_config)
caption_data = training_set.caption_data
token_count = training_set.get_token_count()
concreteness_dataset = generate_concreteness_dataset()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

# model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'
# model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100_extra_18_epochs'
model_name = 'del_me'
# model_name = 'simclr_non_pretrained_noun_th_0.03_conc_num_100'

text_model_dir = os.path.join(models_dir, text_dir)
model = TextualCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
metric = ConcretenessPredictionMetric(model, concreteness_dataset, token_count)
log_print(function_name, 1, metric.report())

log_print(function_name, 0, 'Finished testing')

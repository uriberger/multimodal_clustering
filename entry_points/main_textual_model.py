import os
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Executors
from executors.evaluate_textual_model import TextualModelEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_textual_model'
os.mkdir(timestamp)
# set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'COCO'
dataset_creator, slice_str, multi_label = create_dataset_builder(dataset_name)

test_set_config = DatasetConfig(1, slice_str=slice_str, include_gt_classes=True)
if multi_label:
    test_set, gt_classes_file, _ = dataset_creator.build_dataset(test_set_config)
else:
    test_set = dataset_creator.build_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'

evaluator = TextualModelEvaluator(test_set, model_name, 1)
evaluator.evaluate()
log_print(function_name, 0, 'Finished testing')

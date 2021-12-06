import os
import torch
from utils.general_utils import log_print, models_dir, visual_dir, text_dir, init_entry_point

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Evaluator
from executors.common_evaluator import CommonEvaluator


function_name = 'main_heatmap_evaluation'
timestamp = init_entry_point(True)

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'COCO'
coco_builder, slice_str, multi_label = create_dataset_builder(dataset_name)

test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set, gt_classes_file_path, gt_bboxes_file_path = coco_builder.build_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'

visual_model_dir = os.path.join(models_dir, visual_dir)
text_model_dir = os.path.join(models_dir, text_dir)
evaluator = CommonEvaluator(visual_model_dir, text_model_dir, model_name,
                            test_set, gt_classes_file_path, gt_bboxes_file_path,
                            None, None, 1, metric_list=['heatmap_metric'], batch_size=1)
evaluator.evaluate()

log_print(function_name, 0, 'Finished testing')

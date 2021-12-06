# General
import os
from utils.general_utils import log_print, init_entry_point
from datetime import datetime

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.trainers.train_instance_discrimination import InstanceDiscriminationTrainer


function_name = 'main_instance_discrimination'
timestamp = init_entry_point(True)

log_print(function_name, 0, 'Generating dataset files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

training_set_config = DatasetConfig(1, use_transformations=True)
training_set = coco.build_image_only_dataset(training_set_config)
test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set = coco.build_image_only_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Training model...')
trainer = InstanceDiscriminationTrainer(training_set, 25, 1)
trainer.train()
log_print(function_name, 0, 'Finished training model')

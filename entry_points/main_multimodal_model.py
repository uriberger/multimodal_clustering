# General
import os
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_config import ModelConfig

# Executors
from executors.trainers.train_multimodal_model import MultimodalModelTrainer
# from executors.evaluate_joint_model import JointModelEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_joint_model'
os.mkdir(timestamp)
set_write_to_log(timestamp)

model_config = ModelConfig(
    visual_model='resnet18',
    text_model='lstm',
    cluster_num=100,
    noun_threshold=0.52,
    object_threshold=0.51,
    visual_learning_rate=5e-5,
    pretrained_visual_base_model=True
)
log_print(function_name, 0, str(model_config))

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

training_set_config = DatasetConfig(1)
training_set, _, _ = coco.build_dataset(training_set_config)
test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set, gt_classes_file_path, gt_bboxes_file_path = coco.build_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Training model...')
trainer = MultimodalModelTrainer(timestamp, training_set, 2, model_config, 1)
trainer.train()
log_print(function_name, 0, 'Finished training model')

# log_print(function_name, 0, 'Testing models...')
# evaluator = JointModelEvaluator(timestamp, test_set, gt_classes_file_path, gt_bboxes_file_path, 1)
# evaluator.evaluate()
# log_print(function_name, 0, 'Finished testing model')

# General
import os
from utils.general_utils import log_print, set_write_to_log, visual_dir, text_dir, default_model_name, get_timestamp_str

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_config import ModelConfig

# Executors
from executors.trainers.train_joint_model import JointModelTrainer
from executors.bimodal_evaluators.evaluate_joint_model import JointModelEvaluator


timestamp = get_timestamp_str()
function_name = 'main_joint_model'
os.mkdir(timestamp)
set_write_to_log(timestamp)

model_config = ModelConfig(
    visual_model='resnet50',
    text_model='counts_generative',
    concept_num=100,
    object_threshold=0.5,
    noun_threshold=0.03,
    pretrained_visual_base_model=False,
    freeze_parameters=False
)
log_print(function_name, 0, str(model_config))

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco_builder = Coco(coco_dir, 1)

coco_training_set_config = DatasetConfig(1)
training_set, _, _ = coco_builder.build_dataset(coco_training_set_config)
class_mapping = coco_builder.get_class_mapping()
token_count = training_set.get_token_count()

test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set, gt_classes_file_path, gt_bboxes_file_path = coco_builder.build_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Training model...')
trainer = JointModelTrainer(timestamp, training_set, 2, model_config, None, 1)
# trainer = JointModelTrainer(timestamp, training_set, 2, model_config,
#                             [test_set, gt_classes_file_path, gt_bboxes_file_path, class_mapping, token_count], 1)
trainer.train()
log_print(function_name, 0, 'Finished training model')

log_print(function_name, 0, 'Testing models...')
visual_model_dir = os.path.join(timestamp, visual_dir)
text_model_dir = os.path.join(timestamp, text_dir)
evaluator = JointModelEvaluator(visual_model_dir, text_model_dir, default_model_name, test_set,
                                gt_classes_file_path, gt_bboxes_file_path, class_mapping, token_count,
                                True, 1)
# evaluator = JointModelEvaluator('models/visual', 'models/text', 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100',
#                                 test_set, gt_classes_file_path, gt_bboxes_file_path, class_mapping, token_count,
#                                 True, 1)
evaluator.evaluate()
log_print(function_name, 0, 'Finished testing model')

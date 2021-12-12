# General
import os
from utils.general_utils import log_print, init_entry_point, default_model_name

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_configs.cluster_model_config import ClusterModelConfig

# Executors
from executors.trainers.train_noun_identifier_from_golden import NounIdentifierTrainer
from executors.bimodal_evaluators.evaluate_noun_identifier_from_golden import NounIdentifierEvaluator


function_name = 'main_noun_identifier'
timestamp = init_entry_point(True)

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

training_set_config = DatasetConfig(1)
training_set, _, _ = coco.build_dataset(training_set_config)
test_set_config = DatasetConfig(1, slice_str='val')
test_set, gt_classes_file_path, gt_bboxes_file_path = coco.build_dataset(test_set_config)

cluster_num = 65
log_print(function_name, 0, 'Datasets generated')

config = ClusterModelConfig(
    text_model='generative',
    cluster_num=cluster_num,
    text_threshold=(1/cluster_num + 1/(10*cluster_num))
)
log_print(function_name, 0, str(config))

log_print(function_name, 0, 'Training model...')
trainer = NounIdentifierTrainer(timestamp, training_set, config)
trainer.train()
log_print(function_name, 0, 'Finished training model')

log_print(function_name, 0, 'Testing models...')
evaluator = NounIdentifierEvaluator(timestamp, default_model_name, test_set, gt_classes_file_path, gt_bboxes_file_path,
                                    config, 1)
evaluator.evaluate()
log_print(function_name, 0, 'Finished testing model')

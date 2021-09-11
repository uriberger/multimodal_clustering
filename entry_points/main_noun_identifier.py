# General
import os
from utils.general_utils import log_print, set_write_to_log, default_model_name
from datetime import datetime

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_config import ModelConfig

# Executors
from executors.trainers.train_noun_identifier_from_golden import NounIdentifierTrainer
from executors.bimodal_evaluators.evaluate_noun_identifier_from_golden import NounIdentifierEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_noun_identifier'
os.mkdir(timestamp)
set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

training_set_config = DatasetConfig(1)
training_set, _, _ = coco.build_dataset(training_set_config)
test_set_config = DatasetConfig(1, slice_str='val')
test_set, gt_classes_file_path, gt_bboxes_file_path = coco.build_dataset(test_set_config)

concept_num = 65
log_print(function_name, 0, 'Datasets generated')

config = ModelConfig(
    text_model='counts_generative',
    concept_num=concept_num,
    noun_threshold=(1/concept_num + 1/(10*concept_num))
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

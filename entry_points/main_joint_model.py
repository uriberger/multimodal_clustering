# General
import os
from utils.general_utils import log_print, init_entry_point, project_root_dir

# Dataset
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_configs.cluster_model_config import ClusterModelConfig

# Executors
from executors.trainers.multimodal_clustering_model_trainer import MultimodalClusteringModelTrainer


function_name = 'main_joint_model'
timestamp = init_entry_point(True)

model_config = ClusterModelConfig()
log_print(function_name, 0, str(model_config))

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'COCO'
dataset_builder, _, _ = create_dataset_builder(dataset_name)

training_set_config = DatasetConfig(1)
training_set, _, _ = dataset_builder.build_dataset(training_set_config)
class_mapping = dataset_builder.get_class_mapping()
token_count = training_set.get_token_count()

test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set, gt_classes_file_path, gt_bboxes_file_path = dataset_builder.build_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Training model...')
model_root_dir = os.path.join(project_root_dir, timestamp)
# trainer = MultimodalClusteringModelTrainer(model_root_dir, training_set, 2, model_config, None, 1)
trainer = MultimodalClusteringModelTrainer(model_root_dir, training_set, 2, model_config,
                                           [test_set, gt_classes_file_path, gt_bboxes_file_path, class_mapping,
                                            token_count], 1)
trainer.train()
log_print(function_name, 0, 'Finished training model')

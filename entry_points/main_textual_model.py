from utils.general_utils import log_print, init_entry_point

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Executors
from executors.evaluators.evaluate_textual_model import TextualModelEvaluator


function_name = 'main_textual_model'
init_entry_point(True)

log_print(function_name, 0, 'Generating dataset_files...')
dataset_name = 'COCO'
coco_builder, slice_str, multi_label = create_dataset_builder(dataset_name)

training_set_config = DatasetConfig(1)
training_set, _, _ = coco_builder.build_dataset(training_set_config)
caption_data = training_set.caption_data
token_count = training_set.get_token_count()
test_set_config = DatasetConfig(1, slice_str=slice_str, include_gt_classes=True)
if multi_label:
    test_set, gt_classes_file, _ = coco_builder.build_dataset(test_set_config)
else:
    test_set = coco_builder.build_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing...')

# model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'
model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100_extra_18_epochs'
# model_name = 'simclr_non_pretrained_noun_th_0.03_conc_num_100'

evaluator = TextualModelEvaluator(test_set, model_name, token_count, 1)
evaluator.evaluate()
log_print(function_name, 0, 'Finished testing')

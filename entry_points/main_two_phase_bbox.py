# General
import os
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.two_phase_evaluator import TwoPhaseBBoxEvaluator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_two_phase_bbox'
os.mkdir(timestamp)
# set_write_to_log(timestamp)

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=False,
                                include_gt_bboxes=True, use_transformations=False)
test_set, _, gt_bboxes_file_path = coco.build_image_only_dataset(test_set_config)
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Testing models...')
evaluator = TwoPhaseBBoxEvaluator(test_set, gt_bboxes_file_path, 1, 'resnet_non_pretrained_noun_th_0.03_conc_num_100')
evaluator.evaluate()
log_print(function_name, 0, 'Finished testing model')

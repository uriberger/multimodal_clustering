# General
import os
from utils.general_utils import log_print, init_entry_point

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.demonstrators.bbox_demonstrator import BboxDemonstrator


function_name = 'main_bbox_demonstration'
timestamp = init_entry_point(True)

model_name = 'resnet_pretrained_noun_th_0.06_conc_num_65'

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set, gt_classes_file_path, gt_bboxes_file_path = coco.build_dataset(test_set_config)
class_mapping = coco.get_class_mapping()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Demonstrating bounding boxes...')
demonstrator = BboxDemonstrator(model_name, test_set, gt_classes_file_path, gt_bboxes_file_path,
                                class_mapping, 3, 1)
demonstrator.demonstrate()

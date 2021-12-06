# General
import os
from utils.general_utils import log_print, init_entry_point

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.demonstrators.heatmap_demonstrator import HeatmapDemonstrator


function_name = 'main_heatmap_demonstration'
timestamp = init_entry_point(True)

model_name = 'resnet_50_non_pretrained_noun_th_0.03_conc_num_100'

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set, _, _ = coco.build_dataset(test_set_config)
class_mapping = coco.get_class_mapping()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Demonstrating heatmaps...')
demonstrator = HeatmapDemonstrator(model_name, test_set, class_mapping, 3, 1)
demonstrator.demonstrate()

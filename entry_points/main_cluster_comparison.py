# General
import os
from utils.general_utils import log_print, set_write_to_log
from datetime import datetime

# Dataset
from dataset_builders.coco import Coco
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.clustering_comparison import ClusterComparator


timestamp = str(datetime.now()).replace(' ', '_')
function_name = 'main_cluster_comparison'
os.mkdir(timestamp)
set_write_to_log(timestamp)

model_name = 'resnet_pretrained_noun_th_0.06_conc_num_65'

log_print(function_name, 0, 'Generating dataset_files...')
coco_dir = os.path.join('..', 'datasets', 'COCO')
coco = Coco(coco_dir, 1)

test_set_config = DatasetConfig(1, slice_str='val', include_gt_classes=True, include_gt_bboxes=True)
test_set = coco.build_image_only_dataset(test_set_config)
class_mapping = coco.get_class_mapping()
log_print(function_name, 0, 'Datasets generated')

log_print(function_name, 0, 'Training clusters...')
comparator = ClusterComparator(test_set, 1, model_name)
comparator.train()

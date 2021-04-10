from train_cam_from_golden import train_cam, test_classification
from coco import generate_bboxes_dataset_coco
from cam_model import CAMNet
from cam_dataset import MultiLabelDataset
from aux_functions import log_print, set_write_to_log
from datetime import datetime
import os
import sys
import torch


pretrained_model_path = None
if '--pretrained_model' in sys.argv:
    assert sys.argv[1] == '--pretrained_model'
    pretrained_model_path = sys.argv[2]

timestamp = str(datetime.now()).replace(' ', '_')
os.mkdir(timestamp)
set_write_to_log(timestamp)

log_print('Main', 0, 'Generating datasets...')
img_bboxes_training_set, \
    img_bboxes_test_set, \
    class_ind_to_str = generate_bboxes_dataset_coco()

class_num = len(class_ind_to_str)
model = CAMNet(class_num, pretrained_raw_net=True)
model.to(model.device)
wanted_image_size = (224, 224)

training_set = MultiLabelDataset(img_bboxes_training_set, class_num, wanted_image_size, 'train')
test_set = MultiLabelDataset(img_bboxes_test_set, class_num, wanted_image_size, 'val')
log_print('Main', 0, 'Datasets generated')

if pretrained_model_path is None:
    log_print('Main', 0, 'Training model...')
    train_cam(timestamp, training_set, model, 5)
    log_print('Main', 0, 'Finished training model')
else:
    log_print('Main', 0, 'Loading pretrained model...')
    model.load_state_dict(torch.load(pretrained_model_path))
    log_print('Main', 0, 'Pretrained model loaded')

log_print('Main', 0, 'Testing model...')
test_classification(test_set, model, conf_threshold=0.3)
log_print('Main', 0, 'Finished testing model')

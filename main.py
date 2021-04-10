from train_cam_from_golden import train_cam
from coco import generate_bboxes_dataset_coco
from cam_model import CAMNet
from cam_dataset import MultiLabelDataset
from aux_functions import log_print, set_write_to_log
from datetime import datetime
import os


timestamp = str(datetime.now()).replace(' ', '_')
os.mkdir(timestamp)
set_write_to_log(timestamp)

log_print('Main', 0, 'Generating datasets...')
img_bboxes_training_set, \
    img_bboxes_test_set, \
    class_ind_to_str = generate_bboxes_dataset_coco()

# CHANGE
img_bboxes_training_set = {x: img_bboxes_training_set[x] for x in list(img_bboxes_training_set.keys())[:10]}
img_bboxes_test_set = {x: img_bboxes_test_set[x] for x in list(img_bboxes_test_set.keys())[:2]}
# END CHANGE

class_num = len(class_ind_to_str)
model = CAMNet(class_num, pretrained_raw_net=True)
model.to(model.device)
wanted_image_size = (224, 224)

training_set = MultiLabelDataset(img_bboxes_training_set, class_num, wanted_image_size, 'train')
test_set = MultiLabelDataset(img_bboxes_test_set, class_num, wanted_image_size, 'val')
log_print('Main', 0, 'Datasets generated')

log_print('Main', 0, 'Training model...')
train_cam(timestamp, training_set, test_set, model, 1)
log_print('Main', 0, 'Finished training model')

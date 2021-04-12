from coco import img_caption_training_set_filename, img_caption_val_set_filename
from aux_functions import log_print, set_write_to_log
from datetime import datetime
from train_joint_model import train_joint_model
import os
from captions_dataset import ImageCaptionDataset


timestamp = str(datetime.now()).replace(' ', '_')
os.mkdir(timestamp)
set_write_to_log(timestamp)

log_print('Main', 0, 'Generating datasets...')
wanted_image_size = (224, 224)

training_set = ImageCaptionDataset(wanted_image_size, img_caption_training_set_filename, 'train')
test_set = ImageCaptionDataset(wanted_image_size, img_caption_val_set_filename, 'val')
log_print('Main', 0, 'Datasets generated')

log_print('Main', 0, 'Training model...')
train_joint_model(training_set, 100, 1)
log_print('Main', 0, 'Finished training model')

# log_print('Main', 0, 'Testing model...')
# test_classification(test_set, model, conf_threshold=0.3)
# log_print('Main', 0, 'Finished testing model')

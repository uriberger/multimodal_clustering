from coco import img_caption_training_set_filename, img_caption_val_set_filename, generate_bboxes_dataset_coco
from aux_functions import log_print, set_write_to_log
from datetime import datetime
from train_joint_model import train_joint_model, test_models
import os
from captions_dataset import ImageCaptionDataset
from config import Config


timestamp = str(datetime.now()).replace(' ', '_')
os.mkdir(timestamp)
set_write_to_log(timestamp)

config = Config(
    image_model='resnet18',
    text_model_mode='generative',
    lambda_diversity_loss=0,
    class_num=2
)
log_print('Main', 0, str(config))

log_print('Main', 0, 'Generating datasets...')
wanted_image_size = (224, 224)

# training_set_filename = img_caption_training_set_filename
# test_set_filename = img_caption_val_set_filename
# training_set_filename = 'coco_img_caption_training_set_airplane_bird_single_class'
# test_set_filename = 'coco_img_caption_val_set_airplane_bird_single_class'
training_set_filename = 'coco_img_caption_training_set_airplane_bird_single_class_simplified'
test_set_filename = 'coco_img_caption_val_set_airplane_bird_single_class_simplified'

training_set = ImageCaptionDataset(wanted_image_size, training_set_filename, 'train')
test_set = ImageCaptionDataset(wanted_image_size, test_set_filename, 'val')
log_print('Main', 0, 'Datasets generated')

log_print('Main', 0, 'Training model...')
train_joint_model(timestamp, training_set, 2, 5, config)
log_print('Main', 0, 'Finished training model')

log_print('Main', 0, 'Testing models...')
test_models(timestamp, test_set, 2, config)
log_print('Main', 0, 'Finished testing model')

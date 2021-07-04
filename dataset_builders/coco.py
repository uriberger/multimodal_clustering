import os
import torch
import json
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.visual_utils import get_image_shape_from_id
from utils.text_utils import multiple_word_string
from dataset_builders.dataset_builder import DatasetBuilder


class Coco(DatasetBuilder):

    def __init__(self, root_dir_path, indent):
        super(Coco, self).__init__('coco', indent)
        self.root_dir_path = root_dir_path

        self.train_val_annotations_dir = 'train_val_annotations2014'
        self.test_annotations_dir = 'test_annotations2014'

        self.train_bboxes_filepath_suffix = 'instances_train2014.json'
        self.train_bboxes_filepath = os.path.join(root_dir_path, self.train_val_annotations_dir,
                                                  self.train_bboxes_filepath_suffix)
        self.val_bboxes_filepath_suffix = 'instances_val2014.json'
        self.val_bboxes_filepath = os.path.join(root_dir_path, self.train_val_annotations_dir,
                                                self.val_bboxes_filepath_suffix)
        self.test_bboxes_filepath_suffix = 'image_info_test2014.json'
        self.test_bboxes_filepath = os.path.join(root_dir_path, self.test_annotations_dir,
                                                 self.test_bboxes_filepath_suffix)

        self.train_captions_filepath_suffix = os.path.join(self.train_val_annotations_dir, 'captions_train2014.json')
        self.train_captions_filepath = os.path.join(root_dir_path, self.train_captions_filepath_suffix)
        self.val_captions_filepath_suffix = os.path.join(self.train_val_annotations_dir, 'captions_val2014.json')
        self.val_captions_filepath = os.path.join(root_dir_path, self.val_captions_filepath_suffix)

        self.train_images_dirpath = os.path.join(root_dir_path, 'train2014')
        self.val_images_dirpath = os.path.join(root_dir_path, 'val2014')
        self.test_images_dirpath = os.path.join(root_dir_path, 'test2014')

    def generate_caption_data(self, slice_str):
        return generate_dataset(self.file_paths[slice_str]['captions'], self.generate_caption_data_internal, slice_str)

    def generate_caption_data_internal(self, slice_str):
        if slice_str == 'train':
            external_caption_filepath = self.train_captions_filepath
        elif slice_str == 'val':
            external_caption_filepath = self.val_captions_filepath
        caption_fp = open(external_caption_filepath, 'r')
        caption_data = json.load(caption_fp)
        return caption_data['annotations']

    def generate_gt_classes_data(self, slice_str):
        self.generate_gt_classes_bboxes_data(slice_str)

    def generate_gt_bboxes_data(self, slice_str):
        self.generate_gt_classes_bboxes_data(slice_str)

    def generate_gt_classes_bboxes_data(self, slice_str):
        gt_classes_filepath = self.file_paths[slice_str]['gt_classes']
        gt_bboxes_filepath = self.file_paths[slice_str]['gt_bboxes']

        if os.path.exists(gt_classes_filepath):
            return torch.load(gt_classes_filepath), torch.load(gt_bboxes_filepath)
        else:
            if slice_str == 'train':
                external_bboxes_filepath = self.train_bboxes_filepath
            elif slice_str == 'val':
                external_bboxes_filepath = self.val_bboxes_filepath
            bboxes_fp = open(external_bboxes_filepath, 'r')
            bboxes_data = json.load(bboxes_fp)

            category_id_to_class_id = {bboxes_data[u'categories'][x][u'id']: x for x in
                                       range(len(bboxes_data[u'categories']))}

            img_classes_dataset = {}
            img_bboxes_dataset = {}
            for bbox_annotation in bboxes_data[u'annotations']:
                image_id = bbox_annotation[u'image_id']
                if image_id not in img_bboxes_dataset:
                    img_classes_dataset[image_id] = []
                    img_bboxes_dataset[image_id] = []

                bbox = bbox_annotation[u'bbox']
                xmin = int(bbox[0])
                xmax = int(bbox[0] + bbox[2])
                ymin = int(bbox[1])
                ymax = int(bbox[1] + bbox[3])
                trnsltd_bbox = [xmin, ymin, xmax, ymax]

                category_id = bbox_annotation[u'category_id']
                class_id = category_id_to_class_id[category_id]

                img_classes_dataset[image_id].append(class_id)
                img_bboxes_dataset[image_id].append(trnsltd_bbox)

            torch.save(img_classes_dataset, gt_classes_filepath)
            torch.save(img_bboxes_dataset, gt_bboxes_filepath)

            return img_classes_dataset, img_bboxes_dataset

    def find_unwanted_images(self, slice_str):
        """ We want to filter images that are:
                - Grayscale
                - Contain multiple-words-named classes"""
        img_classes_dataset, _ = self.generate_gt_classes_bboxes_data(slice_str)
        class_mapping = self.get_class_mapping()

        multiple_word_classes = [x for x in class_mapping.keys() if multiple_word_string(class_mapping[x])]
        self.log_print('Multiple word classes: ' + str([class_mapping[x] for x in multiple_word_classes]))

        self.unwanted_images_info = {
            'multiple_word_classes': multiple_word_classes,
            'grayscale_count': 0,
            'multiple_word_classes_count': 0,
            'unwanted_image_ids': [],
            'slice_str': slice_str
        }

        self.increment_indent()
        for_loop_with_reports(img_classes_dataset.items(), len(img_classes_dataset.items()),
                              10000, self.is_unwanted_image, self.unwanted_images_progress_report)
        self.decrement_indent()

        self.log_print('Out of ' + str(len(img_classes_dataset)) + ' images:')
        self.log_print('Found ' + str(self.unwanted_images_info['grayscale_count']) + ' grayscale images')
        self.log_print('Found ' + str(self.unwanted_images_info['multiple_word_classes_count']) +
                       ' multiple word classes')

        return self.unwanted_images_info['unwanted_image_ids']

    def is_unwanted_image(self, index, item, print_info):
        image_id = item[0]
        gt_classes = item[1]

        # Grayscale
        image_shape = get_image_shape_from_id(image_id, self.get_image_path, self.unwanted_images_info['slice_str'])
        if len(image_shape) == 2:
            # Grayscale images only has 2 dims
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['grayscale_count'] += 1
            return

        # Multiple word classes
        if len(set(gt_classes).intersection(self.unwanted_images_info['multiple_word_classes'])) > 0:
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['multiple_word_classes_count'] += 1
            return

    def unwanted_images_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting image ' + str(index) +
                       ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))

    def filter_unwanted_images(self, slice_str):
        """ We want to filter images that are:
        - Grayscale
        - Contain multiple-words-named classes"""
        unwanted_image_ids = self.find_unwanted_images(slice_str)

        caption_dataset = self.generate_caption_data(slice_str)
        img_classes_dataset, img_bboxes_dataset = self.generate_gt_classes_bboxes_data(slice_str)

        new_caption_dataset = [x for x in caption_dataset if x['image_id'] not in unwanted_image_ids]
        new_img_classes_dataset = {x: img_classes_dataset[x] for x in img_classes_dataset.keys()
                                   if x not in unwanted_image_ids}
        new_img_bboxes_dataset = {x: img_bboxes_dataset[x] for x in img_bboxes_dataset.keys()
                                  if x not in unwanted_image_ids}

        torch.save(new_caption_dataset, self.file_paths[slice_str]['captions'])
        torch.save(new_img_classes_dataset, self.file_paths[slice_str]['gt_classes'])
        torch.save(new_img_bboxes_dataset, self.file_paths[slice_str]['gt_bboxes'])

    def get_class_mapping(self):
        bbox_fp = open(self.train_bboxes_filepath, 'r')
        bbox_data = json.load(bbox_fp)

        category_id_to_class_id = {bbox_data[u'categories'][x][u'id']: x for x in range(len(bbox_data[u'categories']))}
        category_id_to_name = {x[u'id']: x[u'name'] for x in bbox_data[u'categories']}
        class_mapping = {category_id_to_class_id[x]: category_id_to_name[x] for x in category_id_to_class_id.keys()}

        return class_mapping

    def get_image_path(self, image_id, slice_str):
        image_filename = 'COCO_' + slice_str + '2014_000000' + '{0:06d}'.format(image_id) + '.jpg'
        if slice_str == 'train':
            images_dirpath = self.train_images_dirpath
        elif slice_str == 'val':
            images_dirpath = self.val_images_dirpath
        elif slice_str == 'test':
            images_dirpath = self.test_images_dirpath
        image_path = os.path.join(images_dirpath, image_filename)

        return image_path

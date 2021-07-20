import abc
import os
from utils.general_utils import generate_dataset
import torch
from loggable_object import LoggableObject
from datasets_src.img_captions_dataset import ImageCaptionDataset
from datasets_src.img_dataset import ImageDataset


class DatasetBuilder(LoggableObject):
    """ This class is the base class for all external datasets. """

    def __init__(self, name, indent):
        super(DatasetBuilder, self).__init__(indent)
        self.name = name

        self.slices = ['train', 'val']
        self.file_paths = {}
        for slice_str in self.slices:
            self.file_paths[slice_str] = self.get_filepaths_for_slice(slice_str)

    def get_filepaths_for_slice(self, slice_str):
        cached_dataset_files_dir = os.path.join('cached_dataset_files')

        return {
            'captions': os.path.join(cached_dataset_files_dir, self.name + '_captions_' + slice_str),
            'gt_classes': os.path.join(cached_dataset_files_dir, self.name + '_gt_classes_' + slice_str),
            'gt_bboxes': os.path.join(cached_dataset_files_dir, self.name + '_gt_bboxes_' + slice_str)
        }

    @abc.abstractmethod
    def generate_caption_data(self, slice_str):
        return

    @abc.abstractmethod
    def generate_gt_classes_data(self, slice_str):
        return

    @abc.abstractmethod
    def generate_gt_bboxes_data(self, slice_str):
        return

    @abc.abstractmethod
    def get_class_mapping(self):
        return

    @abc.abstractmethod
    def get_image_path(self, image_id, slice_str):
        return

    def build_dataset(self, config):
        if config.slice_str not in self.slices:
            self.log_print('No such data slice: ' + str(config.slice_str) +
                           '. Please specify one of ' + str(self.slices))
            assert False

        file_paths = self.file_paths[config.slice_str]

        self.generate_caption_data(config.slice_str)
        if config.include_gt_classes:
            self.generate_gt_classes_data(config.slice_str)
            gt_classes_file_path = file_paths['gt_classes']
        else:
            gt_classes_file_path = None
        if config.include_gt_bboxes:
            self.generate_gt_bboxes_data(config.slice_str)
            gt_bboxes_file_path = file_paths['gt_bboxes']
        else:
            gt_bboxes_file_path = None

        class_mapping = self.get_class_mapping()

        return \
            ImageCaptionDataset(file_paths['captions'],
                                gt_classes_file_path,
                                class_mapping,
                                self.get_image_path,
                                config), \
            gt_classes_file_path, \
            gt_bboxes_file_path

    def build_image_only_dataset(self, config):
        if config.slice_str not in self.slices:
            self.log_print('No such data slice: ' + str(config.slice_str) +
                           '. Please specify one of ' + str(self.slices))
            assert False

        file_paths = self.file_paths[config.slice_str]

        self.generate_caption_data(config.slice_str)

        return ImageDataset(file_paths['captions'], self.get_image_path, config)

    def build_selected_class_caption_dataset(self, class_list, only_single_class_images, slice_str):
        """ This function returns a dataset of (image, caption) pairs, where the dataset only contains
         images with classes of the given class list.
         If only_single_class_images=True, we filter images that contains more than a single class. """
        _, _, class_mapping = self.generate_bboxes_dataset()

        if only_single_class_images:
            single_or_multi_str = 'single'
        else:
            single_or_multi_str = 'multi'

        if slice_str == 'train':
            dataset_filename_prefix = self.img_caption_training_set_filename
        if slice_str == 'val':
            dataset_filename_prefix = self.img_caption_val_set_filename

        dataset_filename = \
            dataset_filename_prefix + \
            '_' + '_'.join([class_mapping[x] for x in class_list]) + \
            '_' + single_or_multi_str + '_class'

        return generate_dataset(dataset_filename, self.build_selected_class_caption_dataset_internal,
                                class_list, only_single_class_images, slice_str)

    def build_selected_class_caption_dataset_internal(self, class_list, only_single_class_images, slice_str):
        self.log_print('Generating dataset...')
        img_bboxes_training_set, img_bboxes_val_set, class_mapping = self.generate_bboxes_dataset()

        if slice_str == 'train':
            full_dataset_filename = self.img_caption_training_set_filename
            boxes_dataset = img_bboxes_training_set
        if slice_str == 'val':
            full_dataset_filename = self.img_caption_val_set_filename
            boxes_dataset = img_bboxes_val_set
        full_dataset = torch.load(full_dataset_filename)

        img_class_dataset = {x[0]: [y[1] for y in x[1]] for x in boxes_dataset.items()}
        if only_single_class_images:
            img_class_dataset = {x[0]: x[1] for x in img_class_dataset.items() if len(x[1]) == 1}

        class_set = set(class_list)
        selected_dataset = [{'image_id': x['image_id'],
                             'caption': x['caption'],
                             'gt_classes': img_class_dataset[x['image_id']]}
                            for x in full_dataset
                            if x['image_id'] in img_class_dataset and
                            len(class_set.intersection(img_class_dataset[x['image_id']])) ==
                            len(img_class_dataset[x['image_id']])]

        class_names = [class_mapping[x] for x in class_list]
        if only_single_class_images:
            at_least_exactly_str = 'exactly'
        else:
            at_least_exactly_str = 'at least'
        self.log_print('Found ' + str(len(selected_dataset)) + ' captions for images containing ' +
                       at_least_exactly_str + ' one of the classes in ' + str(class_names))

        return selected_dataset

    def build_simplified_caption_dataset(self, orig_dataset, filename):
        """ orig_dataset is a dataset of (image, caption) pairs. This function returns a simplified
         dataset of (image, class name). """
        return generate_dataset(filename, self.build_simplified_caption_dataset_internal, orig_dataset)

    def build_simplified_caption_dataset_internal(self, orig_dataset):
        self.log_print('Generating dataset...')
        _, _, class_mapping = self.generate_bboxes_dataset()

        simplified_dataset_with_repetitions = \
            [{'image_id': x['image_id'],
              'caption': ' '.join([class_mapping[y] for y in x['gt_classes']]),
              'gt_classes': x['gt_classes']}
             for x in orig_dataset]
        ''' Since in the original dataset each sample has multiple captions, in the simplified dataset
        we'll have repetitions. So we ant only a unique sample of each image id. '''
        observed_image_ids = {}
        simplified_dataset = []
        for sample in simplified_dataset_with_repetitions:
            image_id = sample['image_id']
            if image_id not in observed_image_ids:
                observed_image_ids[image_id] = True
                simplified_dataset.append(sample)

        return simplified_dataset

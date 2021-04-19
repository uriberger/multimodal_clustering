import os
import time
import json
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from aux_functions import generate_dataset
from selective_search import selective_search

coco_dir = os.path.join('..', 'datasets', 'COCO')
train_val_annotations_dir = 'train_val_annotations2014'
test_annotations_dir = 'test_annotations2014'

train_bboxes_filepath_suffix = 'instances_train2014.json'
train_bboxes_filepath = os.path.join(coco_dir, train_val_annotations_dir, train_bboxes_filepath_suffix)
val_bboxes_filepath_suffix = 'instances_val2014.json'
val_bboxes_filepath = os.path.join(coco_dir, train_val_annotations_dir, val_bboxes_filepath_suffix)
test_bboxes_filepath_suffix = 'image_info_test2014.json'
test_bboxes_filepath = os.path.join(coco_dir, test_annotations_dir, test_bboxes_filepath_suffix)

train_captions_filepath_suffix = os.path.join(train_val_annotations_dir, 'captions_train2014.json')
train_captions_filepath = os.path.join(coco_dir, train_captions_filepath_suffix)
val_captions_filepath_suffix = os.path.join(train_val_annotations_dir, 'captions_val2014.json')
val_captions_filepath = os.path.join(coco_dir, val_captions_filepath_suffix)

train_images_dirpath = os.path.join(coco_dir, 'train2014')
val_images_dirpath = os.path.join(coco_dir, 'val2014')
test_images_dirpath = os.path.join(coco_dir, 'test2014')

img_bboxes_training_set_filename = 'coco_img_bboxes_training_set'
img_bboxes_val_set_filename = 'coco_img_bboxes_val_set'
img_bboxes_test_set_filename = 'coco_img_bboxes_test_set'

img_caption_training_set_filename = 'coco_img_caption_training_set'
img_caption_val_set_filename = 'coco_img_caption_val_set'

img_rps_dataset_filename = 'coco_img_rps_dataset'


def generate_proposals_dataset_coco():
    """ This function returns a dataset of image id -> list of region proposals. """
    return generate_dataset(img_rps_dataset_filename, generate_proposals_dataset_coco_internal)


def generate_proposals_dataset_coco_internal():
    img_rps_temp_filename = 'tmp_rps'
    old_img_rps_temp_filename = 'old' + img_rps_temp_filename
    if os.path.isfile(img_rps_temp_filename):
        img_rps_dataset, img_name_list, cur_file_ind = torch.load(img_rps_temp_filename)
    else:
        img_rps_dataset = {}
        for _, _, files in os.walk(train_images_dirpath):
            img_name_list = files
        cur_file_ind = 0

    file_num = len(img_name_list)
    batch_start_time = time.time()
    while cur_file_ind < file_num:
        if cur_file_ind % 10 == 0:
            batch_time = time.time() - batch_start_time
            print('Calculating region proposals for image ' + str(cur_file_ind) + ' out of ' + str(file_num) +
                  ', prev batch took ' + str(batch_time))
            batch_start_time = time.time()

            # Save results so far
            if os.path.isfile(img_rps_temp_filename):
                os.rename(img_rps_temp_filename, old_img_rps_temp_filename)
            torch.save([img_rps_dataset, img_name_list, cur_file_ind], img_rps_temp_filename)

        image_filename = img_name_list[cur_file_ind]
        image_id = get_image_id(image_filename)
        image_path = get_image_path(image_id)
        image_obj = Image.open(image_path)
        np_image = np.array(np.asarray(image_obj))
        # Check that this image isn't black-and-white, i.e. it has channels
        if len(np_image.shape) == 3:
            region_proposals = selective_search(np_image, mode='fast', random_sort=False)
            img_rps_dataset[image_id] = region_proposals

        cur_file_ind += 1

    return img_rps_dataset


def generate_bboxes_dataset_coco():
    """ This function returns two data structures:
        - A map of image id -> list of bbox, category id pairs (the actual dataset).
        - A category id -> name mapping, for mapping the labels to category names.
    In COCO, we'll use the val set as test set, since the test set has no annotations. """
    training_set, class_mapping = generate_dataset(img_bboxes_training_set_filename,
                                                   generate_bboxes_dataset_coco_internal, 'train')
    val_set, _ = generate_dataset(img_bboxes_val_set_filename, generate_bboxes_dataset_coco_internal, 'val')
    test_set = val_set

    return training_set, test_set, class_mapping


def generate_bboxes_dataset_coco_internal(slice_str):
    # Bboxes data
    if slice_str == 'train':
        bbox_fp = open(train_bboxes_filepath, 'r')
    elif slice_str == 'val':
        bbox_fp = open(val_bboxes_filepath, 'r')
    elif slice_str == 'test':
        bbox_fp = open(test_bboxes_filepath, 'r')
    else:
        print('No such slice \'' + slice_str + '\'')
        assert False

    bbox_data = json.load(bbox_fp)
    bbox_num = len(bbox_data[u'annotations'])

    print('Found ' + str(bbox_num) + ' bboxes')

    category_id_to_class_id = {bbox_data[u'categories'][x][u'id']: x for x in range(len(bbox_data[u'categories']))}
    category_id_to_name = {x[u'id']: x[u'name'] for x in bbox_data[u'categories']}
    class_ind_to_str = {category_id_to_class_id[x]: category_id_to_name[x] for x in category_id_to_class_id.keys()}

    img_bboxes_dataset = {}
    for bbox_annotation in bbox_data[u'annotations']:
        image_id = bbox_annotation[u'image_id']
        if image_id not in img_bboxes_dataset:
            img_bboxes_dataset[image_id] = []

        bbox = bbox_annotation[u'bbox']
        xmin = int(bbox[0])
        xmax = int(bbox[0] + bbox[2])
        ymin = int(bbox[1])
        ymax = int(bbox[1] + bbox[3])
        trnsltd_bbox = [xmin, ymin, xmax, ymax]

        category_id = bbox_annotation[u'category_id']
        class_id = category_id_to_class_id[category_id]
        img_bboxes_dataset[image_id].append((trnsltd_bbox, class_id))

    return img_bboxes_dataset, class_ind_to_str


def get_image_path(image_id, slice_str):
    image_filename = 'COCO_' + slice_str + '2014_000000' + '{0:06d}'.format(image_id) + '.jpg'
    if slice_str == 'train':
        images_dirpath = train_images_dirpath
    elif slice_str == 'val':
        images_dirpath = val_images_dirpath
    elif slice_str == 'test':
        images_dirpath = test_images_dirpath
    image_path = os.path.join(images_dirpath, image_filename)

    return image_path


def get_image_id(image_filename):
    image_name_suffix = image_filename.split('_')[-1]
    image_id_with_leading_zeros = image_name_suffix.split('.')[0]
    image_id = int(image_id_with_leading_zeros)

    return image_id


def generate_selected_class_caption_dataset(class_list, only_single_class_images, slice_str):
    """ This function returns a dataset of (image, caption) pairs, where the dataset only contains
     images with classes of the given class list.
     If only_single_class_images=True, we filter images that contains more than a single class. """
    _, _, class_mapping = generate_bboxes_dataset_coco()

    if only_single_class_images:
        single_or_multi_str = 'single'
    else:
        single_or_multi_str = 'multi'

    if slice_str == 'train':
        dataset_filename_prefix = img_caption_training_set_filename
    if slice_str == 'val':
        dataset_filename_prefix = img_caption_val_set_filename

    dataset_filename = \
        dataset_filename_prefix + \
        '_' + '_'.join([class_mapping[x] for x in class_list]) + \
        '_' + single_or_multi_str + '_class'

    return generate_dataset(dataset_filename, generate_selected_class_caption_dataset_internal,
                            class_list, only_single_class_images, slice_str)


def generate_selected_class_caption_dataset_internal(class_list, only_single_class_images, slice_str):
    print('Generating dataset...')
    img_bboxes_training_set, img_bboxes_val_set, class_mapping = generate_bboxes_dataset_coco()

    if slice_str == 'train':
        full_dataset_filename = img_caption_training_set_filename
        boxes_dataset = img_bboxes_training_set
    if slice_str == 'val':
        full_dataset_filename = img_caption_val_set_filename
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
                        len(class_set.intersection(img_class_dataset[x['image_id']])) > 0]

    class_names = [class_mapping[x] for x in class_list]
    if only_single_class_images:
        at_least_exactly_str = 'exactly'
    else:
        at_least_exactly_str = 'at least'
    print('Found ' + str(len(selected_dataset)) + ' captions for images containing ' +
          at_least_exactly_str + ' one of the classes in ' + str(class_names))

    return selected_dataset


def generate_simplified_caption_dataset(orig_dataset, filename):
    """ orig_dataset is a dataset of (image, caption) pairs. This function returns a simplified
     dataset of (image, class name). """
    return generate_dataset(filename, generate_simplified_caption_dataset_internal, orig_dataset)


def generate_simplified_caption_dataset_internal(orig_dataset):
    print('Generating dataset...')
    _, _, class_mapping = generate_bboxes_dataset_coco()

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

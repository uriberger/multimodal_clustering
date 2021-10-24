import os
from dataset_builders.cifar import Cifar10, Cifar100
from dataset_builders.imagenet import ImageNet
from dataset_builders.coco import Coco
from dataset_builders.flickr30 import Flickr30
from dataset_builders.pascal_voc import PascalVOC
from dataset_builders.wiki_scenes import WikiScenes


def create_dataset_builder(dataset_name):
    root_dir = os.path.join('..', 'datasets', dataset_name)
    if dataset_name == 'cifar-10':
        dataset_generator = Cifar10(root_dir, 1)
        val_slice_str = 'test'
        multi_label = False
    elif dataset_name == 'cifar100':
        dataset_generator = Cifar100(root_dir, 1)
        val_slice_str = 'test'
        multi_label = False
    elif dataset_name == 'ImageNet':
        dataset_generator = ImageNet(root_dir, 1)
        val_slice_str = 'val'
        multi_label = False
    elif dataset_name == 'COCO':
        dataset_generator = Coco(root_dir, 1)
        val_slice_str = 'val'
        multi_label = True
    elif dataset_name == 'flickr30':
        dataset_generator = Flickr30(root_dir, 1)
        val_slice_str = 'val'
        multi_label = True
    elif dataset_name == 'VOC2012':
        dataset_generator = PascalVOC(root_dir, 1)
        val_slice_str = ''
        multi_label = True
    elif dataset_name == 'WikiScenes':
        dataset_generator = WikiScenes(root_dir, 1)
        val_slice_str = ''
        multi_label = True
    else:
        assert False

    return dataset_generator, val_slice_str, multi_label

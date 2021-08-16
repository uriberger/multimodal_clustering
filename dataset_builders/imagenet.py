from utils.visual_utils import pil_image_trans
import torchvision.datasets as datasets
from dataset_builders.dataset_builder import DatasetBuilder


class MyImageNet(datasets.ImageNet):
    def __getitem__(self, idx):
        """ Imagenet original implementation returns list of two items: the first is the images, and the second is
        the labels. To fit our other datasets, we want a mapping instead of a list. """
        batch_list = datasets.ImageNet.__getitem__(self, idx)
        return {
            'image': batch_list[0],
            'label': batch_list[1]
        }


class ImageNet(DatasetBuilder):

    def __init__(self, root_dir_path, indent):
        super(ImageNet, self).__init__(indent)

        self.root_dir_path = root_dir_path
        self.slices = ['train', 'val']

    def get_class_mapping(self):
        class_mapping = {self.imagenet_dataset.class_to_idx[x]: x for x in self.imagenet_dataset.class_to_idx.keys()}

        return class_mapping

    def build_dataset(self, config):
        if config.slice_str not in self.slices:
            self.log_print('No such data slice: ' + str(config.slice_str) +
                           '. Please specify one of ' + str(self.slices))
            assert False

        imagenet_dataset = MyImageNet(self.root_dir_path, split='val', transform=pil_image_trans)
        self.imagenet_dataset = imagenet_dataset

        return imagenet_dataset

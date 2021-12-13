###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from loggable_object import LoggableObject
import os
from utils.general_utils import project_root_dir


# The datasets are assumed to be located in a sibling directory named 'datasets'
datasets_dir = os.path.join(project_root_dir, '..', 'datasets')

# This is the directory in which we will keep the cached files of the datasets we create
cached_dataset_files_dir = os.path.join(project_root_dir, 'cached_dataset_files')


class DatasetBuilder(LoggableObject):
    """ This class is the base class for all external datasets builders.
        A dataset builder is an object that given an external dataset (e.g., an image directory and a file with matching
        captions builds a torch.utils.data.Data object (the actual dataset).
    """

    def __init__(self, indent):
        super(DatasetBuilder, self).__init__(indent)

        self.cached_dataset_files_dir = cached_dataset_files_dir

    """ Get a mapping from class index to class name, for classes labeled in the dataset. """

    @abc.abstractmethod
    def get_class_mapping(self):
        return

    """ Build the torch.utils.data.Data object (the actual dataset) given the configuration provided. """

    @abc.abstractmethod
    def build_dataset(self, config):
        return

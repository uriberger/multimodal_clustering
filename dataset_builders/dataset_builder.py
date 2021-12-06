import abc
from loggable_object import LoggableObject
import os
from utils.general_utils import project_root_dir


datasets_dir = os.path.join(project_root_dir, '..', 'datasets')
cached_dataset_files_dir = os.path.join(project_root_dir, 'cached_dataset_files')


class DatasetBuilder(LoggableObject):
    """ This class is the base class for all external datasets. """
    def __init__(self, indent):
        super(DatasetBuilder, self).__init__(indent)

        self.cached_dataset_files_dir = cached_dataset_files_dir

    @abc.abstractmethod
    def get_class_mapping(self):
        return

    @abc.abstractmethod
    def build_dataset(self, config):
        return

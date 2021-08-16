import abc
from loggable_object import LoggableObject


class DatasetBuilder(LoggableObject):
    """ This class is the base class for all external datasets. """
    def __init__(self, indent):
        super(DatasetBuilder, self).__init__(indent)

    @abc.abstractmethod
    def get_class_mapping(self):
        return

    @abc.abstractmethod
    def build_dataset(self, config):
        return

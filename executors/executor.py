import torch
from models_src.model_config import models_dir
from loggable_object import LoggableObject


class Executor(LoggableObject):
    """ A general class for all executors (trainer, evaluator, etc.) to
    inherit from. """

    def __init__(self, indent):
        super(Executor, self).__init__(indent)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.models_dir = models_dir

    def progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting batch ' + str(index) +
                       ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))

import torch.utils.data as data
from utils.general_utils import for_loop_with_reports
from executors.executor import Executor
import abc


class Trainer(Executor):

    def __init__(self, training_set, epoch_num, batch_size, indent):
        super().__init__(indent)

        self.epoch_num = epoch_num
        self.training_set = training_set
        self.batch_size = batch_size

    @abc.abstractmethod
    def pre_training(self):
        return

    @abc.abstractmethod
    def post_training(self):
        return

    def train(self):
        self.pre_training()

        for epoch_ind in range(self.epoch_num):
            self.log_print('Starting epoch ' + str(epoch_ind))

            dataloader = data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
            # dataloader = data.DataLoader(training_set, batch_size=5, shuffle=True)

            checkpoint_len = 50
            self.increment_indent()
            for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                                  self.train_on_batch, self.progress_report)
            self.decrement_indent()

        self.post_training()

    @abc.abstractmethod
    def train_on_batch(self, index, sampled_batch, print_info):
        return

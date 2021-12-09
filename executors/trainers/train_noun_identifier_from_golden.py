from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper
from trainer import Trainer
from utils.general_utils import default_model_name


class NounIdentifierTrainer(Trainer):

    def __init__(self, timestamp, training_set, config, indent):
        super().__init__(training_set, 1, 1, indent)

        self.model = TextCountsModelWrapper(self.device, config, timestamp, default_model_name, indent + 1)

    def train_on_batch(self, index, sampled_batch, print_info):
        caption = sampled_batch['caption']
        token_list = self.training_set.prepare_data(caption)[0]
        cluster_inds = sampled_batch[1]

        self.model.inference(token_list)
        self.model.training_step(token_list, cluster_inds)

    def pre_training(self):
        self.log_print('Training noun identifier from golden dataset...')

    def post_training(self):
        self.log_print('Finished training noun identifier')

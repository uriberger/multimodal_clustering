from models_src.textual_model_wrapper import TextualCountsModelWrapper
from trainer import Trainer
from utils.text_utils import prepare_data


class NounIdentifierTrainer(Trainer):

    def __init__(self, timestamp, training_set, config, indent):
        super().__init__(training_set, 1, 1, indent)

        self.model = TextualCountsModelWrapper(self.device, config, timestamp, indent+1)

    def train_on_batch(self, index, sampled_batch, print_info):
        caption = sampled_batch['caption']
        token_list = prepare_data(caption)[0]
        concept_inds = sampled_batch[1]

        self.model.inference(token_list)
        self.model.training_step(token_list, concept_inds)

    def pre_training(self):
        self.log_print('Training noun identifier from golden dataset...')

    def post_training(self):
        self.log_print('Finished training noun identifier')

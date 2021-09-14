from executors.trainers.trainer import Trainer
from models_src.multimodal_model_wrapper import MultimodalModelWrapper


class MultimodalModelTrainer(Trainer):

    def __init__(self, timestamp, training_set, epoch_num, config, indent):
        super().__init__(training_set, epoch_num, 50, indent)

        self.model = MultimodalModelWrapper(self.device, config, timestamp, indent + 1)

    def dump_model(self):
        self.model.dump()

    def pre_training(self):
        self.dump_model()

    def post_training(self):
        self.dump_model()

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        captions = sampled_batch['caption']
        token_lists = self.training_set.prepare_data(captions)

        self.model.inference(image_tensor, token_lists)
        if print_info:
            self.log_print(self.model.print_info_on_inference())

        self.model.training_step()

        if print_info:
            self.log_print(self.model.print_info_on_loss())

from trainer import Trainer
from models_src.visual_model_wrapper import VisualModelWrapper


class CamTrainer(Trainer):

    def __init__(self, timestamp, training_set, epoch_num, config, indent):
        super().__init__(training_set, epoch_num, 100, indent)

        self.model = VisualModelWrapper(self.device, config, timestamp, config.visual_model, indent + 1)

    def train_on_batch(self, index, sampled_batch, print_info):
        image_tensor = sampled_batch['image'].to(self.device)
        label_tensor = sampled_batch['label'].to(self.device)

        self.model.inference(image_tensor)
        self.model.training_step(image_tensor, label_tensor)

    def pre_training(self):
        return

    def post_training(self):
        return

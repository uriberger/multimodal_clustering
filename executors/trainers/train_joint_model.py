from utils.text_utils import prepare_data
import torch
from executors.trainers.trainer import Trainer
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import generate_textual_model


class JointModelTrainer(Trainer):

    def __init__(self, timestamp, training_set, epoch_num, config, indent):
        super().__init__(training_set, epoch_num, 100, indent)

        self.visual_model = VisualModelWrapper(self.device, config, timestamp, indent+1)
        self.text_model = generate_textual_model(self.device, config, timestamp, indent + 1)

    def dump_models(self):
        self.visual_model.dump()
        self.text_model.dump()

    def pre_training(self):
        self.dump_models()

    def post_training(self):
        self.dump_models()

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        captions = sampled_batch['caption']
        batch_size = len(captions)
        token_lists = prepare_data(captions)

        # Infer
        self.visual_model.inference(image_tensor)
        if print_info:
            self.log_print(self.visual_model.print_info_on_inference())
        self.text_model.inference(token_lists)

        # Train text model, assuming that the visual model is already trained
        # 1. Use visual model for inference
        labels_by_visual = self.visual_model.predict_concept_indicators()
        if print_info:
            predictions_num = sum([torch.sum(labels_by_visual[i]) for i in range(batch_size)])
            self.log_print('Predicted ' + str(int(predictions_num.item())) + ' concepts according to visual')

        # 2. Use the result to train textual model
        self.text_model.training_step(token_lists, labels_by_visual)

        # Train visual model, assuming that the text model is already trained
        # 1. Use textual model for inference
        labels_by_text = self.text_model.predict_concept_indicators()
        if print_info:
            self.log_print(self.text_model.print_info_on_inference())

        # 2. Use the result to train visual model
        self.visual_model.training_step(image_tensor, labels_by_text)
        if print_info:
            self.log_print(self.visual_model.print_info_on_loss())

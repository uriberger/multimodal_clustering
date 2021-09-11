import os
from utils.general_utils import visual_dir, text_dir, default_model_name
from utils.text_utils import prepare_data
import torch
from executors.trainers.trainer import Trainer
from executors.bimodal_evaluators.evaluate_joint_model import JointModelEvaluator
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import generate_textual_model


class JointModelTrainer(Trainer):

    def __init__(self, timestamp, training_set, epoch_num, config, test_data, indent):
        super().__init__(training_set, epoch_num, 100, indent)

        self.visual_model_dir = os.path.join(timestamp, visual_dir)
        self.text_model_dir = os.path.join(timestamp, text_dir)
        self.visual_model = VisualModelWrapper(self.device, config, self.visual_model_dir,
                                               default_model_name, indent + 1)
        self.text_model = generate_textual_model(self.device, config, self.text_model_dir,
                                                 default_model_name, indent + 1)

        self.visual_loss_history = []
        self.text_loss_history = []

        self.prev_checkpoint_batch_ind = 0
        self.test_data = test_data

    def dump_models(self):
        self.visual_model.dump()
        self.text_model.dump()

    def pre_training(self):
        self.dump_models()

    def post_training(self):
        self.dump_models()

    def post_loop(self):
        self.dump_models()
        if self.test_data is not None:
            # If test data was provided, we evaluate after every epoch
            evaluator = JointModelEvaluator(self.visual_model_dir, self.text_model_dir, default_model_name,
                                            self.test_data[0], self.test_data[1], self.test_data[2], self.test_data[3],
                                            False, self.indent + 1)
            evaluator.evaluate()

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

        # Document loss
        loss = self.text_model.cached_loss + self.visual_model.cached_loss
        self.loss_history.append(loss)
        self.text_loss_history.append(self.text_model.cached_loss)
        self.visual_loss_history.append(self.visual_model.cached_loss)
        if print_info:
            batch_count_from_prev_checkpoint = len(self.loss_history) - self.prev_checkpoint_batch_ind
            mean_loss = \
                sum(self.loss_history[self.prev_checkpoint_batch_ind:])/batch_count_from_prev_checkpoint
            mean_text_loss = \
                sum(self.text_loss_history[self.prev_checkpoint_batch_ind:]) / batch_count_from_prev_checkpoint
            mean_visual_loss = \
                sum(self.visual_loss_history[self.prev_checkpoint_batch_ind:]) / batch_count_from_prev_checkpoint
            self.log_print('Mean loss: ' + str(mean_loss) +
                           ', mean text loss: ' + str(mean_text_loss) +
                           ', mean visual loss: ' + str(mean_visual_loss))
            self.prev_checkpoint_batch_ind = len(self.loss_history)

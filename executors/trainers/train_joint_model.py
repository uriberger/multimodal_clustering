import os
import csv
from utils.general_utils import visual_dir, text_dir, default_model_name
import torch
from executors.trainers.trainer import Trainer
from executors.common_evaluator import CommonEvaluator
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import generate_textual_model


class JointModelTrainer(Trainer):

    def __init__(self, timestamp, training_set, epoch_num, config, test_data, indent,
                 loaded_model_dir=None, loaded_model_name=None):
        super().__init__(training_set, epoch_num, 50, indent)

        self.timestamp = timestamp
        if loaded_model_dir is None:
            self.visual_model_dir = os.path.join(timestamp, visual_dir)
            self.text_model_dir = os.path.join(timestamp, text_dir)
            os.mkdir(self.visual_model_dir)
            os.mkdir(self.text_model_dir)
            self.model_name = default_model_name
            visual_config = config
            textual_config = config
        else:
            self.visual_model_dir = os.path.join(loaded_model_dir, visual_dir)
            self.text_model_dir = os.path.join(loaded_model_dir, text_dir)
            self.model_name = loaded_model_name
            visual_config = None
            textual_config = config.text_model
        self.visual_model = VisualModelWrapper(self.device, visual_config, self.visual_model_dir,
                                               self.model_name, indent + 1)
        self.text_model = generate_textual_model(self.device, textual_config, self.text_model_dir,
                                                 self.model_name, indent + 1)

        self.visual_loss_history = []
        self.text_loss_history = []

        self.prev_checkpoint_batch_ind = 0
        self.test_data = test_data
        self.evaluation_results = []

    def dump_models(self):
        self.visual_model.dump()
        self.text_model.dump()

    def pre_training(self):
        self.dump_models()

        self.first_epoch = True

    def post_training(self):
        self.dump_models()

    def generate_metric_to_results_mapping(self):
        metric_to_results = {}
        precision_str = "%.4f"
        for epoch_ind in range(len(self.evaluation_results)):
            epoch_results = self.evaluation_results[epoch_ind]
            for metric_name, metric_result in epoch_results.items():
                if metric_name not in metric_to_results:
                    metric_to_results[metric_name] = {'metric': metric_name}
                if isinstance(metric_result, float) or isinstance(metric_result, int):
                    result_str = precision_str % metric_result
                elif isinstance(metric_result, tuple):
                    result_str = str([precision_str % x for x in metric_result])
                else:
                    # Not implemented
                    assert False
                metric_to_results[metric_name][str(epoch_ind)] = result_str

        return metric_to_results

    def evaluate_current_model(self):
        evaluator = CommonEvaluator(self.visual_model_dir, self.text_model_dir, self.model_name,
                                    self.test_data[0], self.test_data[1], self.test_data[2], self.test_data[3],
                                    self.test_data[4], self.indent + 1)
        results = evaluator.evaluate()
        self.evaluation_results.append(results)

    def dump_results_to_csv(self):
        csv_filename = 'evaluation_by_epoch.csv'
        if self.first_epoch:
            csv_filename = 'first_' + csv_filename

        with open(os.path.join(self.timestamp, csv_filename), 'w', newline='') as csvfile:
            step_num = len(self.evaluation_results)
            fieldnames = ['metric'] + [str(x) for x in range(step_num)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            metric_to_results = self.generate_metric_to_results_mapping()
            for result_dic in metric_to_results.values():
                writer.writerow(result_dic)

    def post_epoch(self):
        self.dump_models()

        if self.test_data is not None:
            # First dump results of first epoch (we record every few batches)
            if self.first_epoch:
                self.dump_results_to_csv()
                self.evaluation_results = []
                self.first_epoch = False

            self.log_print('Evaluating after finishing the epoch...')
            # If test data was provided, we evaluate after every epoch
            self.evaluate_current_model()

            # Dump results into a csv file
            self.dump_results_to_csv()

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        captions = sampled_batch['caption']
        batch_size = len(captions)
        token_lists = self.training_set.prepare_data(captions)

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

        if print_info and self.test_data is not None and self.first_epoch:
            self.evaluate_current_model()

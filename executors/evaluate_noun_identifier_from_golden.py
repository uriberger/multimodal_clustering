from evaluator import Evaluator
import torch.utils.data as data
import metrics
import os
from models_src.textual_model_wrapper import TextualCountsModelWrapper


class NounIdentifierEvaluator(Evaluator):

    def __init__(self, timestamp, test_set, gt_classes_file_path, gt_bboxes_file_path,
                 config, indent, model_name=None):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        # Load models
        if model_name is not None:
            model_dir = os.path.join(self.models_dir, 'text')
        else:
            model_dir = timestamp

        self.model = TextualCountsModelWrapper(self.device, config, model_dir, model_name)

    def evaluate(self):
        self.log_print('Evaluating noun identifier...')
        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=True)
        metric_list = [
            metrics.NounIdentificationMetric(self.model, self.img_bboxes_val_set)
        ]
        self.run_metrics_on_dataset(metric_list, dataloader)
        self.log_print('Finished evaluating noun identifier')

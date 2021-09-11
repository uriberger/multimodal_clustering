from bimodal_evaluator import BimodalEvaluator
import torch.utils.data as data
import metrics
from models_src.textual_model_wrapper import TextualCountsModelWrapper


class NounIdentifierEvaluator(BimodalEvaluator):

    def __init__(self, model_dir, model_name, test_set, gt_classes_file_path, gt_bboxes_file_path,
                 config, indent):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        self.model = TextualCountsModelWrapper(self.device, config, model_dir, model_name, indent)

    def evaluate(self):
        self.log_print('Evaluating noun identifier...')
        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=True)
        metric_list = [
            metrics.NounIdentificationMetric(self.model, self.img_bboxes_val_set)
        ]
        self.run_metrics_on_dataset(metric_list, dataloader)
        self.log_print('Finished evaluating noun identifier')

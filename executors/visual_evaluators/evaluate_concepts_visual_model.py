import torch
import os
from executors.visual_evaluators.evaluate_visual_model import VisualModelEvaluator
from metrics import VisualUnknownClassesClassificationMetric
from models_src.visual_model_wrapper import VisualModelWrapper


class VisualConceptEvaluator(VisualModelEvaluator):

    def __init__(self, test_set, class_mapping, gt_classes_file_path, model_str, indent):
        if gt_classes_file_path is None:
            self.multi_label = False
        else:
            self.multi_label = True
            self.gt_classes_data = torch.load(gt_classes_file_path)

        super(VisualConceptEvaluator, self).__init__(test_set, class_mapping, 'concepts', model_str, indent)

        self.metric = VisualUnknownClassesClassificationMetric(None)

    def inference(self, inputs):
        self.model.inference(inputs)
        if self.multi_label:
            return torch.tensor(self.model.predict_concept_indicators())
        else:
            return torch.tensor(self.model.predict_classes()).view(1, inputs.shape[0]).transpose(1, 0)

    def generate_model(self, model_type, model_str):
        model_dir = os.path.join(self.models_dir, 'visual')
        model = VisualModelWrapper(self.device, None, model_dir, self.indent + 1, model_str)
        model.eval()

        inference_func = self.inference

        return model, inference_func

    def metric_pre_calculations(self):
        return

    def predict_classes(self, sample_ind):
        if self.multi_label:
            concept_indicators = self.embedding_mat[sample_ind]
            return [x for x in range(self.embedding_mat.shape[1]) if concept_indicators[x] == 1]
        else:
            return [self.embedding_mat[sample_ind].item()]

import torch
import os
from executors.visual_evaluators.evaluate_visual_model import VisualModelEvaluator
from metrics import VisualKnownClassesClassificationMetric
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import TextualCountsModelWrapper
from utils.general_utils import visual_dir, text_dir


class VisualClusterEvaluator(VisualModelEvaluator):

    def __init__(self, test_set, class_mapping, gt_classes_file_path, model_str, indent):
        if gt_classes_file_path is None:
            self.multi_label = False
        else:
            self.multi_label = True
            self.gt_classes_data = torch.load(gt_classes_file_path)

        super(VisualClusterEvaluator, self).__init__(test_set, class_mapping, 'clusters', model_str, indent)

        class_num = len([x for x in self.class_mapping.values() if ' ' not in x])
        self.metric = VisualKnownClassesClassificationMetric(None, class_num)

    def inference(self, inputs):
        self.model.inference(inputs)
        if self.multi_label:
            return self.model.predict_cluster_indicators()
        else:
            return torch.tensor(self.model.predict_classes()).view(1, inputs.shape[0]).transpose(1, 0)

    def generate_model(self, model_type, model_str):
        visual_model_dir = os.path.join(self.models_dir, visual_dir)
        visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_str, self.indent + 1)
        visual_model.eval()

        text_model_dir = os.path.join(self.models_dir, text_dir)
        self.text_model = TextualCountsModelWrapper(self.device, None, text_model_dir, model_str, self.indent + 1)

        inference_func = self.inference

        return visual_model, inference_func

    def metric_pre_calculations(self):
        self.cluster_to_gt_class = self.text_model.create_cluster_to_gt_class_mapping(self.class_mapping)

    def predict_classes(self, sample_ind):
        if self.multi_label:
            cluster_indicators = self.embedding_mat[sample_ind]
            predicted_clusters = [x for x in range(self.embedding_mat.shape[1]) if cluster_indicators[x] == 1]
            predicted_class_lists = [self.cluster_to_gt_class[x] for x in predicted_clusters]
            predicted_classes = [inner for outer in predicted_class_lists for inner in outer]
        else:
            predicted_cluster = self.embedding_mat[sample_ind].item()
            predicted_classes = self.cluster_to_gt_class[predicted_cluster]
        return predicted_classes

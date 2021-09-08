import torch
import os
from executors.visual_evaluators.evaluate_visual_model import VisualModelEvaluator
from metrics import VisualKnownClassesClassificationMetric
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import TextualCountsModelWrapper


class VisualConceptEvaluator(VisualModelEvaluator):

    def __init__(self, test_set, class_mapping, gt_classes_file_path, model_str, indent):
        if gt_classes_file_path is None:
            self.multi_label = False
        else:
            self.multi_label = True
            self.gt_classes_data = torch.load(gt_classes_file_path)

        super(VisualConceptEvaluator, self).__init__(test_set, class_mapping, 'concepts', model_str, indent)

        class_num = len([x for x in self.class_mapping.values() if ' ' not in x])
        self.metric = VisualKnownClassesClassificationMetric(None, class_num)

    def inference(self, inputs):
        self.model.inference(inputs)
        if self.multi_label:
            return self.model.predict_concept_indicators()
        else:
            return torch.tensor(self.model.predict_classes()).view(1, inputs.shape[0]).transpose(1, 0)

    def generate_model(self, model_type, model_str):
        visual_model_dir = os.path.join(self.models_dir, 'visual')
        visual_model = VisualModelWrapper(self.device, None, visual_model_dir, self.indent + 1, model_str)
        visual_model.eval()

        text_model_dir = os.path.join(self.models_dir, 'text')
        self.text_model = TextualCountsModelWrapper(self.device, None, text_model_dir, self.indent + 1, model_str)

        inference_func = self.inference

        return visual_model, inference_func

    def metric_pre_calculations(self):
        gt_class_to_prediction = {i: self.text_model.model.predict_concept(self.class_mapping[i])
                                  for i in self.class_mapping.keys()
                                  if ' ' not in self.class_mapping[i]}
        gt_class_to_concept = {x[0]: x[1][0] for x in gt_class_to_prediction.items() if x[1] is not None}
        concept_num = self.model.config.concept_num
        concept_to_gt_class = {concept_ind: [] for concept_ind in range(concept_num)}
        for gt_class_ind, concept_ind in gt_class_to_concept.items():
            concept_to_gt_class[concept_ind].append(gt_class_ind)

        self.concept_to_gt_class = concept_to_gt_class

    def predict_classes(self, sample_ind):
        if self.multi_label:
            concept_indicators = self.embedding_mat[sample_ind]
            predicted_concepts = [x for x in range(self.embedding_mat.shape[1]) if concept_indicators[x] == 1]
            predicted_class_lists = [self.concept_to_gt_class[x] for x in predicted_concepts]
            predicted_classes = [inner for outer in predicted_class_lists for inner in outer]
        else:
            predicted_concept = self.embedding_mat[sample_ind].item()
            predicted_classes = self.concept_to_gt_class[predicted_concept]
        return predicted_classes

from executors.bimodal_evaluators.bimodal_evaluator import BimodalEvaluator
from dataset_builders.concreteness_dataset import generate_concreteness_dataset
from dataset_builders.category_dataset import generate_category_dataset
import os
import spacy
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import TextualCountsModelWrapper
import torch.utils.data as data
import metrics


class JointModelEvaluator(BimodalEvaluator):

    def __init__(self, timestamp, test_set, gt_classes_file_path, gt_bboxes_file_path, indent, model_name=None):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        # Load datasets
        self.concreteness_dataset = generate_concreteness_dataset()
        self.category_dataset = generate_category_dataset()

        # Load models
        if model_name is not None:
            visual_model_dir = os.path.join(self.models_dir, 'visual')
            text_model_dir = os.path.join(self.models_dir, 'text')
        else:
            visual_model_dir = timestamp
            text_model_dir = timestamp

        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, indent+1, model_name)
        self.visual_model.eval()
        self.text_model = TextualCountsModelWrapper(self.device, None, text_model_dir, indent+1, model_name)

        self.nlp = spacy.load("en_core_web_sm")

    def evaluate(self):
        """ Go over the test set and evaluate using the metrics. """
        self.log_print('Evaluating metrics')
        dataloader = data.DataLoader(self.test_set, batch_size=100, shuffle=True)
        metric_list = [
            metrics.BBoxMetric(self.visual_model),
            metrics.NounIdentificationMetric(self.text_model, self.nlp),
            metrics.ConcretenessPredictionMetric(self.text_model, self.concreteness_dataset),
            metrics.CategorizationMetric(self.text_model, self.category_dataset),
            metrics.VisualUnknownClassesClassificationMetric(self.visual_model),
            metrics.SentenceImageMatchingMetric(self.visual_model, self.text_model)
        ]
        self.run_metrics_on_dataset(metric_list, dataloader)

    def infer(self, visual_metadata, visual_inputs, textual_inputs):
        self.visual_model.inference(visual_inputs)
        self.text_model.inference(textual_inputs)

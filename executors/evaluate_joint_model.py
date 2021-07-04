from executors.evaluator import Evaluator
from dataset_builders.concreteness_dataset import generate_concreteness_dataset
import os
import spacy
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import TextualCountsModelWrapper
import torch.utils.data as data
import metrics


class JointModelEvaluator(Evaluator):

    def __init__(self, timestamp, test_set, gt_classes_file_path, gt_bboxes_file_path, indent, model_name=None):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        # Load datasets
        self.concreteness_dataset = generate_concreteness_dataset()

        # Load models
        if model_name is not None:
            visual_model_dir = os.path.join(self.models_dir, 'visual')
            text_model_dir = os.path.join(self.models_dir, 'text')
        else:
            visual_model_dir = timestamp
            text_model_dir = timestamp

        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, indent+1, model_name)
        self.visual_model.no_grad()
        self.text_model = TextualCountsModelWrapper(self.device, None, text_model_dir, indent+1, model_name)

        self.nlp = spacy.load("en_core_web_sm")

    def evaluate(self):
        """ First, go over the test set sample by sample, and evaluate using
        the single-sample metrics. """
        self.log_print('Evaluating single sample metrics')
        single_dataloader = data.DataLoader(self.test_set, batch_size=2, shuffle=True)
        single_sample_metrics = [
            metrics.BBoxMetric(self.visual_model),
            metrics.NounIdentificationMetric(self.text_model, self.nlp),
            metrics.ConcretenessPredictionMetric(self.text_model, self.concreteness_dataset)
        ]
        self.run_metrics_on_dataset(single_sample_metrics, single_dataloader)

        """ Next, go over the pairs of samples, and evaluate using
        the sample pair metrics. """
        self.log_print('Evaluating sample pair metrics')
        pair_dataloader = data.DataLoader(self.test_set, batch_size=20, shuffle=True)
        sample_pair_metrics = [
            metrics.SentenceImageMatchingMetric(self.visual_model, self.text_model)
        ]
        self.run_metrics_on_dataset(sample_pair_metrics, pair_dataloader)

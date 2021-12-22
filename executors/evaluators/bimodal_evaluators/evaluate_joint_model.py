from executors.evaluators.bimodal_evaluators.bimodal_evaluator import BimodalEvaluator
from dataset_builders.concreteness_dataset import ConcretenessDatasetBuilder
from dataset_builders.category_dataset import CategoryDatasetBuilder
import spacy
from models_src.wrappers.visual_model_wrapper import VisualModelWrapper
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper
import torch.utils.data as data
import metrics


class JointModelEvaluator(BimodalEvaluator):

    def __init__(self, visual_model_dir, text_model_dir, model_name, test_set,
                 gt_classes_file_path, gt_bboxes_file_path, class_mapping, token_count,
                 evaluate_bbox, indent):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)
        self.class_mapping = class_mapping
        self.evaluate_bbox = evaluate_bbox

        # Load datasets
        self.concreteness_dataset = ConcretenessDatasetBuilder(self.indent + 1).build_dataset()
        self.category_dataset = CategoryDatasetBuilder(self.indent + 1).build_dataset()
        self.token_count = token_count

        # Load models
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_name, indent + 1)
        self.visual_model.eval()
        self.text_model = TextCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)

        self.nlp = spacy.load("en_core_web_sm")

    def evaluate(self):
        """ Go over the test set and evaluate using the metrics. """
        self.log_print('Evaluating metrics')
        dataloader = data.DataLoader(self.test_set, batch_size=100, shuffle=True)
        metric_list = []
        if self.evaluate_bbox:
            metric_list.append(metrics.BBoxPredictionMetric(self.visual_model))
        metric_list += [
            metrics.NounIdentificationMetric(self.text_model, self.nlp),
            metrics.ConcretenessPredictionMetric(self.text_model, self.concreteness_dataset, self.token_count),
            metrics.CategorizationMetric(self.text_model, self.category_dataset),
            metrics.VisualUnknownClassesClassificationMetric(self.visual_model, 'co_occur'),
            metrics.VisualUnknownClassesClassificationMetric(self.visual_model, 'iou'),
            metrics.VisualPromptClassificationMetric(self.visual_model, self.text_model, self.class_mapping),
            metrics.SentenceImageMatchingMetric(self.visual_model, self.text_model)
        ]
        results = self.run_metrics_on_dataset(metric_list, dataloader)

        return results

    def infer(self, visual_metadata, visual_inputs, textual_inputs):
        self.visual_model.inference(visual_inputs)
        self.text_model.inference(textual_inputs)

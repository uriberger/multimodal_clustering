###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from executors.evaluators.visual_evaluators.visual_evaluator import VisualEvaluator

# Metrics
from metrics.sensitivity_specificity_metrics.compare_to_gt_bbox_metrics.heatmap_metric import HeatmapMetric

# Models
from models_src.wrappers.visual_model_wrapper import ActivationMapRandomPredictionModelWrapper


class RandomHeatmapEvaluator(VisualEvaluator):
    """ This evaluator evaluates the Heatmap metric using random guesses. """

    def __init__(self, test_set, gt_classes_file_path, gt_bboxes_file_path, indent):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        # Model
        self.model = ActivationMapRandomPredictionModelWrapper()

        self.metrics = [HeatmapMetric(self.model)]

    def infer(self, visual_input, visual_metadata):
        gt_bboxes = visual_metadata['gt_bboxes']
        self.model.set_prediction_num([len(x) for x in gt_bboxes])

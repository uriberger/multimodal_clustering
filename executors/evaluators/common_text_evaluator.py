###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from executors.executor import Executor

# Metrics
from metrics.categorization_metric import CategorizationMetric
from metrics.concreteness_prediction_metric import ConcretenessPredictionMetric
from metrics.cluster_counter_metric import ClusterCounterMetric

# Datasets
from dataset_builders.category_dataset import CategoryDatasetBuilder
from dataset_builders.concreteness_dataset import ConcretenessDatasetBuilder

# Models
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper


class CommonTextEvaluator(Executor):
    """ This evaluator is the most commonly used in our project (after each epoch in the training of multimodal
        clustering model). It evaluates the model on text tasks.
    """

    def __init__(self, text_model_dir, model_name, token_count, indent):
        super().__init__(indent)

        self.token_count = token_count

        # Load Model
        self.model = TextCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)
        self.model.eval()

        # Datasets
        category_dataset = CategoryDatasetBuilder(self.indent + 1).build_dataset()
        concreteness_dataset = ConcretenessDatasetBuilder(self.indent + 1).build_dataset()
        # similarity_dataset = WordSimDatasetBuilder(False, self.indent + 1).build_dataset()
        # relatedness_dataset = WordSimDatasetBuilder(True, self.indent + 1).build_dataset()

        self.metrics = [
            CategorizationMetric(self.model, category_dataset, ignore_unknown_words=True),
            CategorizationMetric(self.model, category_dataset, ignore_unknown_words=False),
            ConcretenessPredictionMetric(self.model, concreteness_dataset, token_count),
            ClusterCounterMetric(self.model, token_count),
            # WordSimilarityMetric(self.model, similarity_dataset, False, True),
            # WordSimilarityMetric(self.model, relatedness_dataset, True, True),
        ]

    """ The entry point of this class. """

    def evaluate(self):
        results = {}
        for metric in self.metrics:
            self.log_print(metric.report())
            results.update(metric.results)
        self.decrement_indent()

        return results

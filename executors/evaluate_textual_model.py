import torch
import torch.utils.data as data
from utils.general_utils import for_loop_with_reports, models_dir
from utils.text_utils import prepare_data
from executors.executor import Executor
from dataset_builders.concreteness_dataset import generate_concreteness_dataset
import os
import spacy
from models_src.textual_model_wrapper import TextualCountsModelWrapper
from metrics import NounIdentificationMetric, ConcretenessPredictionMetric


class TextualModelEvaluator(Executor):
    """ Evaluate a textual pre-trained model for various metrics. """

    def __init__(self, test_set, model_name, indent):
        super(TextualModelEvaluator, self).__init__(indent)

        self.test_set = test_set

        text_model_dir = os.path.join(models_dir, 'text')
        self.model = TextualCountsModelWrapper(self.device, None, text_model_dir, indent + 1, model_name)
        self.concreteness_dataset = generate_concreteness_dataset()
        self.nlp = spacy.load("en_core_web_sm")

    def evaluate(self):
        """ Go over the test set and evaluate using the metrics. """
        self.log_print('Evaluating metrics')
        dataloader = data.DataLoader(self.test_set, batch_size=100, shuffle=True)
        metric_list = [
            NounIdentificationMetric(self.model, self.nlp),
            ConcretenessPredictionMetric(self.model, self.concreteness_dataset)
        ]
        self.run_metrics_on_dataset(metric_list, dataloader)

    def run_metrics_on_dataset(self, metric_list, data_loader):
        self.increment_indent()
        self.metric_list = metric_list
        checkpoint_len = 100
        self.increment_indent()
        for_loop_with_reports(data_loader, len(data_loader), checkpoint_len,
                              self.evaluate_on_batch, self.progress_report)
        self.decrement_indent()
        # Report results
        for metric in metric_list:
            self.log_print(metric.report())
        self.decrement_indent()

    def evaluate_on_batch(self, index, sampled_batch, print_info):
        # Load data
        with torch.no_grad():
            captions = sampled_batch['caption']
            token_lists = prepare_data(captions)

            # Infer
            self.model.inference(token_lists)

            for metric in self.metric_list:
                metric.predict_and_document(None, None, token_lists)

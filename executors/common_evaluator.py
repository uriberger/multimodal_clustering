import torch
import torch.utils.data as data
from executors.executor import Executor
from utils.general_utils import for_loop_with_reports

# Metrics
from metrics import CategorizationMetric, ConcretenessPredictionMetric, VisualPromptClassificationMetric, HeatmapMetric

# Datasets
from dataset_builders.category_dataset import generate_fountain_category_dataset
from dataset_builders.concreteness_dataset import generate_concreteness_dataset

# Models
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import TextualCountsModelWrapper


class CommonEvaluator(Executor):

    def __init__(self, visual_model_dir, text_model_dir, model_name,
                 test_set, gt_classes_file_path, gt_bboxes_file_path,
                 class_mapping, token_count,
                 indent):
        super().__init__(indent)

        self.test_set = test_set
        self.gt_classes_data = torch.load(gt_classes_file_path)
        self.gt_bboxes_data = torch.load(gt_bboxes_file_path)

        # Models
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_name, indent + 1)
        self.visual_model.eval()
        self.text_model = TextualCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)

        # Datasets
        category_dataset = generate_fountain_category_dataset()
        concreteness_dataset = generate_concreteness_dataset()

        # Metrics
        self.metrics = [
            CategorizationMetric(self.text_model, category_dataset, predicted_labels=None, ignore_unknown_words=True),
            CategorizationMetric(self.text_model, category_dataset, predicted_labels=None, ignore_unknown_words=False),
            ConcretenessPredictionMetric(self.text_model, concreteness_dataset, token_count),
            VisualPromptClassificationMetric(self.visual_model, self.text_model, class_mapping),
            HeatmapMetric(self.visual_model)
        ]

        # Determine on which portion of the test set we need to run inference
        self.determine_constraints()

    def determine_constraints(self):
        found_text_metric_on_test_set = False
        found_visual_metric_on_test_set = False
        for metric in self.metrics:
            if metric.is_image_only():
                found_visual_metric_on_test_set = True
            if not metric.uses_external_dataset():
                found_visual_metric_on_test_set = True
                found_text_metric_on_test_set = True

        self.need_to_run_on_test_set = found_visual_metric_on_test_set or found_text_metric_on_test_set
        self.need_to_run_only_on_images = found_visual_metric_on_test_set and (not found_text_metric_on_test_set)
        self.need_to_run_only_on_text = found_text_metric_on_test_set and (not found_visual_metric_on_test_set)
        self.need_to_run_on_both_modalities = found_visual_metric_on_test_set and found_text_metric_on_test_set

    def evaluate(self):
        # Evaluate for each metric on the test set
        if self.need_to_run_on_test_set:
            self.run_metrics_on_test_set()

        # Extract results
        return self.extract_results()

    def run_metrics_on_test_set(self):
        """ Go over the test set and evaluate using the metrics. """
        self.log_print('Evaluating metrics')
        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=False)
        self.visited_image_ids = {}

        self.increment_indent()
        checkpoint_len = 100
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.run_metrics_on_batch, self.progress_report)
        self.decrement_indent()

    def run_metrics_on_batch(self, index, sampled_batch, print_info):
        with torch.no_grad():
            batch_size = len(sampled_batch['image_id'])

            image_ids = [x.item() for x in sampled_batch['image_id']]
            captions = sampled_batch['caption']
            image_tensor = sampled_batch['image'].to(self.device)
            orig_image_size = [(sampled_batch['orig_image_size'][0][i].item(),
                                sampled_batch['orig_image_size'][1][i].item()) for i in range(batch_size)]
            token_lists = self.test_set.prepare_data(captions)

        ''' Ground-truth classes and bboxes are not part of the dataset, because these are lists of varying
        length and thus cannot be batched using pytorch's data loader. So we need to extract these from an
        external file. '''
        gt_classes = [self.gt_classes_data[x] for x in image_ids]
        gt_bboxes = [self.gt_bboxes_data[x] for x in image_ids]

        visual_metadata = {
            'image_id': image_ids,
            'orig_image_size': orig_image_size,
            'gt_classes': gt_classes,
            'gt_bboxes': gt_bboxes
        }

        # Infer
        self.infer(image_ids, image_tensor, token_lists)

        # Calculate metrics
        for metric in self.metrics:
            metric.predict_and_document(visual_metadata, image_tensor, token_lists)

    def infer(self, image_ids, visual_input, text_input):
        batch_size = len(image_ids)
        for sample_ind in range(batch_size):
            image_id = image_ids[sample_ind]

            if self.need_to_run_on_both_modalities or \
                    (self.need_to_run_only_on_images and image_id not in self.visited_image_ids):
                self.visited_image_ids[image_id] = True

                image_tensor = visual_input[[sample_ind], :, :, :]
                self.visual_model.inference(image_tensor)

                if self.need_to_run_on_both_modalities or self.need_to_run_only_on_text:
                    token_list = text_input[sample_ind]
                    self.text_model.inference([token_list])

    def extract_results(self):
        results = {}
        for metric in self.metrics:
            self.log_print(metric.report())
            results.update(metric.results)
        self.decrement_indent()

        return results

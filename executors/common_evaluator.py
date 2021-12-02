import torch
import torch.utils.data as data
from executors.executor import Executor
from utils.general_utils import for_loop_with_reports

# Metrics
from metrics import \
    CategorizationMetric, \
    ConcretenessPredictionMetric, \
    VisualPromptClassificationMetric, \
    ClusterCounterMetric, \
    HeatmapMetric

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
                 indent, metric_list=None, batch_size=25):
        super().__init__(indent)

        self.batch_size = batch_size

        self.test_set = test_set
        self.gt_classes_data = torch.load(gt_classes_file_path)
        self.gt_bboxes_data = torch.load(gt_bboxes_file_path)

        # Models
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_name, indent + 1)
        self.visual_model.eval()
        self.text_model = TextualCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)
        self.text_model.model.calculate_probs()

        # Datasets
        category_dataset = generate_fountain_category_dataset()
        concreteness_dataset = generate_concreteness_dataset()

        # Metrics
        if metric_list is None:
            self.metrics = [
                CategorizationMetric(self.text_model, category_dataset, ignore_unknown_words=True),
                CategorizationMetric(self.text_model, category_dataset, ignore_unknown_words=False),
                ConcretenessPredictionMetric(self.text_model, concreteness_dataset, token_count),
                VisualPromptClassificationMetric(self.visual_model, self.text_model, class_mapping),
                ClusterCounterMetric(self.text_model, token_count)
            ]
        else:
            self.metrics = [self.metric_name_to_object(x) for x in metric_list]

    def metric_name_to_object(self, metric_name):
        if metric_name == 'heatmap_metric':
            return HeatmapMetric(self.visual_model)

    def evaluate(self):
        # Evaluate for each metric on the test set
        self.run_metrics_on_test_set()

        # Extract results
        return self.extract_results()

    def run_metrics_on_test_set(self):
        """ Go over the test set and evaluate using the metrics. """
        self.log_print('Evaluating metrics')
        dataloader = data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        self.visited_image_ids = {}

        self.increment_indent()
        checkpoint_len = 800
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
            'image_ids': image_ids,
            'orig_image_size': orig_image_size,
            'gt_classes': gt_classes,
            'gt_bboxes': gt_bboxes
        }

        # Calculate metrics
        self.calculate_metrics_on_batch(visual_metadata, image_tensor, token_lists)

    def calculate_metrics_on_batch(self, visual_metadata, image_tensor, token_lists):
        image_ids = visual_metadata['image_ids']
        for metric in self.metrics:
            if metric.uses_external_dataset():
                continue

            # We should filter out visited images
            filtered_visual_metadata = visual_metadata
            filtered_image_tensor = image_tensor
            filtered_token_lists = token_lists
            if metric.is_image_only():
                not_visited_image_id_to_index = {image_ids[i]: i for i in range(len(image_ids))
                                                 if image_ids[i] not in self.visited_image_ids}
                not_visited_image_ids_indices = list(not_visited_image_id_to_index.values())
                filtered_visual_metadata = {
                    'image_id': [image_ids[i] for i in not_visited_image_ids_indices],
                    'orig_image_size': [visual_metadata['orig_image_size'][i] for i in not_visited_image_ids_indices],
                    'gt_classes': [visual_metadata['gt_classes'][i] for i in not_visited_image_ids_indices],
                    'gt_bboxes': [visual_metadata['gt_bboxes'][i] for i in not_visited_image_ids_indices]
                }
                filtered_image_tensor = filtered_image_tensor[not_visited_image_ids_indices]
                filtered_token_lists = [filtered_token_lists[i] for i in not_visited_image_ids_indices]

            if len(filtered_token_lists) == 0:
                continue

            # Infer
            self.infer(filtered_image_tensor, filtered_token_lists)

            # Run metric
            metric.predict_and_document(filtered_visual_metadata, filtered_image_tensor, filtered_token_lists)

        for image_id in image_ids:
            self.visited_image_ids[image_id] = True

    def infer(self, visual_input, text_input):
        self.visual_model.inference(visual_input)
        self.text_model.inference(text_input)

    def extract_results(self):
        results = {}
        for metric in self.metrics:
            self.log_print(metric.report())
            results.update(metric.results)
        self.decrement_indent()

        return results

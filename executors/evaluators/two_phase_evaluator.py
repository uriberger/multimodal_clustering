import torch
import torch.utils.data as data
from utils.general_utils import generate_dataset, for_loop_with_reports, visual_dir
from metrics.sensitivity_specificity_metrics.compare_to_gt_bbox_metrics.bbox_prediction_metric \
    import BBoxPredictionMetric
from executors.executor import Executor
import os
from models_src.wrappers.visual_model_wrapper import VisualModelWrapper


class TwoPhaseBBoxEvaluator(Executor):
    """ This is like the normal evaluator, but here there are two separated phases:
    First, we run inference on all samples, and then we calculate the metrics. """

    def __init__(self, test_set, gt_bboxes_file, indent, model_name):
        super().__init__(indent)

        self.test_set = test_set
        self.gt_bboxes_data = torch.load(gt_bboxes_file)

        # Load models
        visual_model_dir = os.path.join(self.models_dir, visual_dir)

        self.model = VisualModelWrapper(self.device, None, visual_model_dir, indent + 1, model_name)
        self.model.eval()

        self.metric = BBoxPredictionMetric(self.model)
        self.recorded_data_path = os.path.join('cached_dataset_files', 'recorded_activation_maps')
        self.recorded_data = []

    def evaluate(self):
        # Run inference on dataset
        self.recorded_data = generate_dataset(self.recorded_data_path, self.run_inference_on_dataset)

        # Calculate the metric
        self.metric_on_dataset()

        # Report results
        self.report_results()

    def run_inference_on_dataset(self):
        self.log_print('Running inference on entire dataset')
        dataloader = data.DataLoader(self.test_set, batch_size=100, shuffle=True)
        checkpoint_len = 10
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.run_inference_on_batch, self.progress_report)
        self.decrement_indent()

        return self.recorded_data

    def run_inference_on_batch(self, index, sampled_batch, print_info):
        # Load data
        with torch.no_grad():
            batch_size = len(sampled_batch['image_id'])

            image_ids = [x.item() for x in sampled_batch['image_id']]
            image_tensor = sampled_batch['image'].to(self.device)
            orig_image_size = [(sampled_batch['orig_image_size'][0][i].item(),
                                sampled_batch['orig_image_size'][1][i].item()) for i in range(batch_size)]

            ''' Ground-truth classes and bboxes are not part of the dataset, because these are lists of varying
            length and thus cannot be batched using pytorch's data loader. So we need to extract these from an
            external file. '''
            gt_bboxes = [self.gt_bboxes_data[x] for x in image_ids]

            visual_metadata = {
                'orig_image_size': orig_image_size,
                'gt_bboxes': gt_bboxes
            }

            # Infer
            self.infer_and_record(visual_metadata, image_tensor)

    def infer_and_record(self, visual_metadata, image_tensor):
        orig_image_size = visual_metadata['orig_image_size']
        batch_size = image_tensor.shape[0]
        gt_bboxes = visual_metadata['gt_bboxes']

        activation_maps = self.model.predict_activation_maps(image_tensor)
        activation_maps = [x[1] for x in activation_maps]
        for sample_ind in range(batch_size):
            cur_record = {
                'orig_image_size': orig_image_size[sample_ind],
                'gt_bboxes': gt_bboxes[sample_ind],
                'activation_maps': activation_maps[sample_ind]
            }
            self.recorded_data.append(cur_record)

    def metric_on_dataset(self):
        self.log_print('Running metric on entire dataset')
        checkpoint_len = 100
        self.increment_indent()
        for_loop_with_reports(self.recorded_data, len(self.recorded_data), checkpoint_len,
                              self.metrics_on_batch, self.progress_report)
        self.decrement_indent()

    def metrics_on_batch(self, index, record, print_info):
        orig_image_size = record['orig_image_size']
        gt_bboxes = record['gt_bboxes']
        activation_maps = record['activation_maps']

        self.metric.document_with_loaded_results([orig_image_size], [activation_maps], [gt_bboxes])

    def report_results(self):
        self.log_print('Results:')
        self.log_print(self.metric.report())

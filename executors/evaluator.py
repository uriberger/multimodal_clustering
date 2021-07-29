import torch
from utils.general_utils import  for_loop_with_reports
from utils.text_utils import prepare_data
from executors.executor import Executor
import abc


class Evaluator(Executor):

    def __init__(self, test_set, gt_classes_file, gt_bboxes_file, indent):
        super().__init__(indent)

        self.test_set = test_set
        self.gt_classes_data = torch.load(gt_classes_file)
        self.gt_bboxes_data = torch.load(gt_bboxes_file)

    @abc.abstractmethod
    def evaluate(self):
        return

    @abc.abstractmethod
    def infer(self, visual_metadata, visual_inputs, textual_inputs):
        return

    def run_metrics_on_dataset(self, metric_list, data_loader):
        self.increment_indent()
        self.metric_list = metric_list
        checkpoint_len = 10
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
            batch_size = len(sampled_batch['image_id'])

            image_ids = [x.item() for x in sampled_batch['image_id']]
            captions = sampled_batch['caption']
            image_tensor = sampled_batch['image'].to(self.device)
            orig_image_size = [(sampled_batch['orig_image_size'][0][i].item(),
                                sampled_batch['orig_image_size'][1][i].item()) for i in range(batch_size)]

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
            token_lists = prepare_data(captions)

            # Infer
            self.infer(visual_metadata, image_tensor, token_lists)

            for metric in self.metric_list:
                metric.predict_and_document(visual_metadata, image_tensor, token_lists)

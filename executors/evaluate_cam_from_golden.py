from evaluator import Evaluator
import torch.utils.data as data
import metrics
import os
from models_src.visual_model_wrapper import VisualModelWrapper


class CamEvaluator(Evaluator):

    def __init__(self, timestamp, test_set, config,
                 generate_bboxes_dataset_func, indent,
                 model_name=None):
        super().__init__(test_set, indent)

        # Load cached_dataset_files
        _, self.img_bboxes_val_set, _ = generate_bboxes_dataset_func()

        # Load models
        if model_name is not None:
            model_dir = os.path.join(self.models_dir, 'visual')
        else:
            model_dir = timestamp

        self.model = VisualModelWrapper(self.device, config, model_dir, model_name, indent+1)
        self.model.no_grad()

    def evaluate(self):
        self.log_print('Evaluating CAM model...')
        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=True)
        metric_list = [
            metrics.BBoxMetric(self.model, self.img_bboxes_val_set),
            metrics.VisualKnownClassesClassificationMetric(self.model)
        ]
        self.run_metrics_on_dataset(metric_list, dataloader)
        self.log_print('Finished evaluating CAM model')

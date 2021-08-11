import torch
import torch.utils.data as data
from utils.general_utils import generate_dataset, for_loop_with_reports
from metrics import VisualUnknownClassesClassificationMetric
from executors.executor import Executor
import os
from models_src.embedding_model_wrapper import EmbeddingModelWrapper
from models_src.clip_wrapper import ClipWrapper
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.model_config import ModelConfig, wanted_image_size


class EmbeddingModelEvaluator(Executor):
    """ Evaluate an embedding pre-trained model for self-supervised clustering. """
    root_dir = 'embedding_models'
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    model_dir = os.path.join(root_dir, 'models')
    embedding_mat_dir = os.path.join(root_dir, 'embedding_mat')

    def __init__(self, test_set, class_mapping, model_name, indent):
        super().__init__(indent)

        self.test_set = test_set

        self.model = self.generate_embedding_model(model_name, len(class_mapping))
        self.model.eval()

        self.metric = VisualUnknownClassesClassificationMetric(None)
        self.embedding_mat_path = os.path.join(self.embedding_mat_dir, model_name)
        if not os.path.isdir(self.embedding_mat_dir):
            os.mkdir(self.embedding_mat_dir)

        # Get embedding dimension
        dummy_image = torch.zeros(1, 3, wanted_image_size[0], wanted_image_size[1])
        dummy_output = self.model.inference(dummy_image)
        embedding_dim = dummy_output.shape[1]

        self.embedding_mat = torch.zeros(len(self.test_set), embedding_dim)
        self.batch_size = 100

        # Get gt classes num
        self.gt_classes_num = len(class_mapping)

    def generate_embedding_model(self, model_name, class_num):
        if model_name == 'pretrained_resnet':
            model_config = ModelConfig(
                visual_model='resnet18',
                concept_num=class_num,
                pretrained_visual_base_model=True
            )
            model = EmbeddingModelWrapper(self.device, model_config, self.model_dir, self.indent + 1, model_name)
        elif model_name.startswith('clip_'):
            specific_clip_model = model_name.split('clip_')[1]
            model_config = ModelConfig(
                visual_model=specific_clip_model
            )
            model = ClipWrapper(self.device, model_config, self.model_dir, self.indent + 1)
        else:
            model_wrapper = VisualModelWrapper(self.device, None, 'visual', self.indent + 1, model_name)

        return model

    def evaluate(self):
        # Create embedding matrix
        embedding_mat = generate_dataset(self.embedding_mat_path, self.run_inference_on_dataset)

        # Cluster
        self.model.cluster(embedding_mat)

        # Calculate the metric
        self.metric_on_dataset()

        # Report results
        self.report_results()

    def run_inference_on_dataset(self):
        self.log_print('Running inference on entire dataset')
        dataloader = data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        checkpoint_len = 10
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.run_inference_on_batch, self.progress_report)
        self.decrement_indent()

        return self.embedding_mat

    def run_inference_on_batch(self, index, sampled_batch, print_info):
        # Load data
        with torch.no_grad():
            image_tensor = sampled_batch['image'].to(self.device)

            batch_start = index*self.batch_size
            batch_end = min((index+1)*self.batch_size, len(self.test_set))

            # Infer
            self.infer_and_record(image_tensor, batch_start, batch_end)

    def infer_and_record(self, image_tensor, batch_start, batch_end):
        output = self.model.inference(image_tensor)
        self.embedding_mat[batch_start:batch_end, :] = output

    def metric_on_dataset(self):
        self.log_print('Running metric on entire dataset')
        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=False)
        checkpoint_len = 10000
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.metrics_on_batch, self.progress_report)
        self.decrement_indent()

    def metrics_on_batch(self, index, sampled_batch, print_info):
        # Load data
        label = sampled_batch['label'].to(self.device)
        predicted_class = self.model.image_id_to_label[index]
        self.metric.document([[predicted_class]], [[label.item()]])

    def report_results(self):
        self.log_print('Results:')
        self.log_print(self.metric.report())

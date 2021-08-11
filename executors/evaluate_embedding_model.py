import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.visual_utils import generate_visual_model
from metrics import VisualUnknownClassesClassificationMetric
from executors.executor import Executor
import os
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.model_config import wanted_image_size
import clip
from sklearn.cluster import KMeans


class EmbeddingModelEvaluator(Executor):
    """ Evaluate an embedding pre-trained model for self-supervised clustering. """
    root_dir = 'embedding_models'
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    model_dir = os.path.join(root_dir, 'models')
    embedding_mat_dir = os.path.join(root_dir, 'embedding_mat')

    def __init__(self, test_set, class_mapping, model_type, model_str, indent):
        super().__init__(indent)

        self.test_set = test_set

        model, inference_func = self.generate_embedding_model(model_type, model_str)
        self.model = model
        self.inference_func = inference_func
        self.model.eval()

        self.metric = VisualUnknownClassesClassificationMetric(None)
        model_name = model_type + '_' + model_str
        self.embedding_mat_path = os.path.join(self.embedding_mat_dir, model_name)
        if not os.path.isdir(self.embedding_mat_dir):
            os.mkdir(self.embedding_mat_dir)

        # Get embedding dimension
        dummy_image = torch.zeros(1, 3, wanted_image_size[0], wanted_image_size[1])
        dummy_output = self.inference_func(dummy_image)
        embedding_dim = dummy_output.shape[1]

        self.embedding_mat = torch.zeros(len(self.test_set), embedding_dim)
        self.batch_size = 100

        # Get gt classes num
        self.gt_classes_num = len(class_mapping)

    def generate_embedding_model(self, model_type, model_str):
        if model_type == 'pretrained':
            model = generate_visual_model(model_str, 1, True)
            model.fc = nn.Identity()
            inference_func = model.forward
        elif model_type == 'clip':
            model, _ = clip.load(model_str, self.device)
            inference_func = model.encode_image
        elif model_type == 'unimodal':
            model_wrapper = VisualModelWrapper(self.device, None, 'models/visual', self.indent + 1, model_str)
            model = model_wrapper.model
            model.fc = nn.Identity()
            inference_func = model.forward
        else:
            # No such model type
            assert False
        return model, inference_func

    def evaluate(self):
        # Create embedding matrix
        embedding_mat = generate_dataset(self.embedding_mat_path, self.create_embedding_mat)

        # Cluster
        self.cluster(embedding_mat)

        # Calculate the metric
        self.metric_on_dataset()

        # Report results
        self.report_results()

    def create_embedding_mat(self):
        self.log_print('Creating embedding matrix')
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
        output = self.inference_func(image_tensor)
        self.embedding_mat[batch_start:batch_end, :] = output

    def cluster(self, embedding_mat, soft_clustering=False):
        if soft_clustering:
            return
        else:
            kmeans = KMeans(n_clusters=self.gt_classes_num).fit(embedding_mat.detach().numpy())
            cluster_list = list(kmeans.labels_)
            self.image_id_to_label = {
                x: cluster_list[x] for x in range(len(cluster_list))
            }

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
        predicted_class = self.image_id_to_label[index]
        self.metric.document([[predicted_class]], [[label.item()]])

    def report_results(self):
        self.log_print('Results:')
        self.log_print(self.metric.report())

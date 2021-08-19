import torch
import torch.nn as nn
from utils.visual_utils import generate_visual_model
from executors.embedding_evaluators.evaluate_embedding_model import EmbeddingModelEvaluator
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.simclr import SimCLRModel, clean_state_dict
import clip
from sklearn.cluster import KMeans
import os
from metrics import VisualUnknownClassesClassificationMetric


class ClusteringEvaluator(EmbeddingModelEvaluator):
    """ Evaluate an embedding pre-trained model for self-supervised classification using clustering. """

    def __init__(self, test_set, class_mapping, model_type, model_str, indent):
        super(ClusteringEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)
        self.metric = VisualUnknownClassesClassificationMetric(None)

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
        elif model_type == 'simclr':
            model = SimCLRModel()
            model_path = os.path.join(EmbeddingModelEvaluator.root_dir, model_str)
            model.load_state_dict(clean_state_dict(torch.load(model_path, map_location=self.device)))
            inference_func = lambda x: model.forward(x)[0]
        else:
            # No such model type
            assert False
        return model, inference_func

    def metric_pre_calculations(self):
        # if self.soft_clustering:
        if False:
            return
        else:
            gt_class_num = len(self.class_mapping)
            sample_num = self.embedding_mat.shape[0]
            with torch.no_grad():
                emb_norm = torch.norm(self.embedding_mat, p=2, dim=1).view(sample_num, 1)
                normalized_embedding_mat = self.embedding_mat / emb_norm
            # kmeans = KMeans(n_clusters=gt_class_num).fit(self.embedding_mat.detach().numpy())
            kmeans = KMeans(n_clusters=gt_class_num).fit(normalized_embedding_mat.detach().numpy())
            cluster_list = list(kmeans.labels_)
            self.image_id_to_label = {
                x: cluster_list[x] for x in range(len(cluster_list))
            }

    def predict_class(self, sample_ind):
        return self.image_id_to_label[sample_ind]

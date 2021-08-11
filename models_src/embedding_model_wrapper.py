import torch
from models_src.model_wrapper import ModelWrapper
from utils.visual_utils import generate_visual_model
from sklearn.cluster import KMeans


class EmbeddingModelWrapper(ModelWrapper):
    """ The purpose of this class is to represent models that are trained to create meaningful visual embeddings, and
    later on uses these embeddings to cluster and classify images in an unsupervised setting. """

    def __init__(self, device, config, model_dir, indent, name):
        super(EmbeddingModelWrapper, self).__init__(device, config, model_dir, indent, name)
        self.cached_output = None

    def generate_model(self):
        return generate_visual_model(self.config.visual_model, self.config.concept_num,
                                     self.config.pretrained_visual_base_model)

    def dump_model(self):
        torch.save(self.model.state_dict(), self.get_model_path())

    def load_model(self):
        self.model.load_state_dict(torch.load(self.get_model_path(), map_location=torch.device(self.device)))

    def eval(self):
        self.model.eval()

    def inference(self, inputs):
        output = self.model(inputs)
        self.cached_output = output
        return output

    def cluster(self, embedding_mat, soft_clustering=False):
        if soft_clustering:
            return
        else:
            kmeans = KMeans(n_clusters=self.config.concept_num).fit(embedding_mat.detach().numpy())
            cluster_list = list(kmeans.labels_)
            self.image_id_to_label = {
                x: cluster_list[x] for x in range(len(cluster_list))
            }

import torch
from executors.evaluators.visual_evaluators.clustering_evaluators.evaluate_clustering import ClusteringEvaluator


class ClusteringSingleLabelEvaluator(ClusteringEvaluator):
    """ Evaluate an embedding pre-trained model for self-supervised classification using clustering for single-label
    tasks. """

    def __init__(self, test_set, class_mapping, model_type, model_str, indent):
        super(ClusteringSingleLabelEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)
        self.multi_label = False

    def metric_pre_calculations(self):
        gt_class_num = len(self.class_mapping)
        sample_num = self.embedding_mat.shape[0]
        with torch.no_grad():
            emb_norm = torch.norm(self.embedding_mat, p=2, dim=1).view(sample_num, 1)
            normalized_embedding_mat = self.embedding_mat / emb_norm
        # kmeans = KMeans(n_clusters=gt_class_num).fit(self.embedding_mat.detach().numpy())
        # kmeans = KMeans(n_clusters=gt_class_num).fit(normalized_embedding_mat.detach().numpy())
        # cluster_list = list(kmeans.labels_)
        # CHANGE
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(n_clusters=gt_class_num, assign_labels='discretize').fit(normalized_embedding_mat.detach().numpy())
        cluster_list = list(sc.labels_)
        # END CHANGE
        self.image_id_to_label = {
            x: cluster_list[x] for x in range(len(cluster_list))
        }

    def predict_classes(self, sample_ind):
        return [self.image_id_to_label[sample_ind]]

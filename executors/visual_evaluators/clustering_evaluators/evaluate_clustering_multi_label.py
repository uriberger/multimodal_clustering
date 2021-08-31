from executors.visual_evaluators.clustering_evaluators.evaluate_clustering import ClusteringEvaluator
from utils.multi_label_threshold_finder import generate_sample_to_predicted_classes_mapping
import torch
import numpy as np
import skfuzzy as fuzz
import torch.utils.data as data


class ClusteringMultiLabelEvaluator(ClusteringEvaluator):
    """ Evaluate an embedding pre-trained model for self-supervised classification using clustering for multi-label
    tasks. """

    def __init__(self, test_set, class_mapping, gt_classes_file, model_type, model_str, indent):
        super(ClusteringMultiLabelEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)

        self.gt_classes_data = torch.load(gt_classes_file)
        self.multi_label = True

    def metric_pre_calculations(self):
        """ In multi-label clustering evaluation, there's a heavy pre-calculation stage.
        First, we need to create fuzzy clusters: to map samples to probability of being included in our made-up
        clusters. """
        self.log_print('Creating fuzzy clusters...')
        self.create_fuzzy_clusters()

        """ Next, we need to map each cluster to which class it represents (many to one mapping). We'll use the
        following heuristic: for each cluster, we'll choose the class with the highest sum of probabilities (over all
        samples). """
        self.log_print('Creating cluster to class mapping...')
        self.map_cluster_to_class()

        """ Next, we calculate the probability over classes for each sample. """
        self.log_print('Collecting prob and gt...')
        self.collect_prob_and_gt()

        """ Finally, we need to choose the probability threshold using the multi-label threshold utility. """
        self.log_print('Generating sample to  predicted class mapping...')
        self.sample_to_predicted_classes = generate_sample_to_predicted_classes_mapping(self.prob_gt_list)

    def create_fuzzy_clusters(self):
        gt_class_num = len(self.class_mapping)
        sample_num = self.embedding_mat.shape[0]
        with torch.no_grad():
            emb_norm = torch.norm(self.embedding_mat, p=2, dim=1).view(sample_num, 1)
            normalized_embedding_mat = self.embedding_mat / emb_norm

        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data=normalized_embedding_mat.detach().numpy().transpose(), c=gt_class_num, m=2, error=0.005, maxiter=1000, init=None)
        self.predicted_cluster_probs = torch.tensor(np.array(u).transpose())

    def map_cluster_to_class(self):
        gt_class_num = len(self.class_mapping)
        cluster_to_class_prob_sum = {
            cluster_ind:
                {
                    class_ind: 0 for class_ind in self.class_mapping.keys()
                }
            for cluster_ind in range(gt_class_num)}

        class_count = {
            class_ind: 0 for class_ind in self.class_mapping.keys()
        }

        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=False)
        for index, batch in enumerate(dataloader):
            gt_classes = self.get_labels_from_batch(batch)
            sample_predicted_probs = self.predicted_cluster_probs[index]
            for cluster_ind in range(gt_class_num):
                for gt_class in gt_classes:
                    cluster_to_class_prob_sum[cluster_ind][gt_class] += sample_predicted_probs[cluster_ind].item()
                    class_count[gt_class] += 1

        normalized_c2c_prob_sum = {
            cluster_ind: {
                class_ind:
                    cluster_to_class_prob_sum[cluster_ind][class_ind]/class_count[class_ind]
                    if class_count[class_ind] > 0 else 0
                for class_ind in self.class_mapping.keys()
            } for cluster_ind in range(gt_class_num)
        }

        cluster_to_class = {
            cluster_ind:
                max(normalized_c2c_prob_sum[cluster_ind], key=normalized_c2c_prob_sum[cluster_ind].get)
            for cluster_ind in range(gt_class_num)
        }

        self.class_to_clusters = {
            class_ind: [] for class_ind in self.class_mapping.keys()
        }
        for cluster_ind, class_ind in cluster_to_class.items():
            self.class_to_clusters[class_ind].append(cluster_ind)

    def collect_prob_and_gt(self):
        self.prob_gt_list = []

        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=False)
        for sample_ind, sampled_batch in enumerate(dataloader):
            predicted_cluster_probs_for_sample = self.predicted_cluster_probs[sample_ind]
            gt_classes = self.get_labels_from_batch(sampled_batch)
            for class_ind in self.class_mapping.keys():
                relevant_clusters = self.class_to_clusters[class_ind]
                class_predicted_prob = torch.sum(predicted_cluster_probs_for_sample[relevant_clusters]).item()
                if class_predicted_prob > 0:
                    self.prob_gt_list.append((class_predicted_prob, class_ind in gt_classes, sample_ind, class_ind))

    def predict_classes(self, sample_ind):
        return self.sample_to_predicted_classes[sample_ind]

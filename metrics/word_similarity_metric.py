###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import numpy as np
from metrics.metric import Metric


class WordSimilarityMetric(Metric):
    """ This metric uses a concreteness dataset: a human-annotated dataset
        that gives every word a number between 1 and 5, representing how
        concrete is this word. We compare it to our prediction.
    """

    def __init__(self, text_model, word_similarity_dataset, relatedness, ignore_unknown):
        super(WordSimilarityMetric, self).__init__(None, text_model)
        self.word_similarity_dataset = word_similarity_dataset
        self.tested_pairs_count = 0

        self.gt_list = []
        self.pred_list = []

        self.same_cluster_sim_sum = 0
        self.same_cluster_count = 0
        self.diff_cluster_sim_sum = 0
        self.diff_cluster_count = 0

        if relatedness:
            self.name = 'relatedness'
        else:
            self.name = 'similarity'

        self.ignore_unknown = ignore_unknown

    """ Go over the entire validation set (words that occurred in our training set, i.e. are in token_count, intersected
        with words that occur in the external concreteness dataset), predict concreteness, and compare to ground-truth.
    """

    def traverse_validation_set(self):
        for word1, word2, gt_similarity in self.word_similarity_dataset:
            # concreteness_prediction = self.text_model.estimate_word_concreteness([[token]])[0][0]
            word1_clusters = self.text_model.underlying_model.get_cluster_conditioned_on_word(word1)
            if word1_clusters is None:
                if self.ignore_unknown:
                    continue
                else:
                    word1_clusters = [1] * self.text_model.config.cluster_num
            word2_clusters = self.text_model.underlying_model.get_cluster_conditioned_on_word(word2)
            if word2_clusters is None:
                if self.ignore_unknown:
                    continue
                else:
                    word2_clusters = [1] * self.text_model.config.cluster_num

            self.tested_pairs_count += 1

            word1_prob_vec = np.array(word1_clusters)
            word2_prob_vec = np.array(word2_clusters)
            similarity_prediction = np.matmul(word1_prob_vec, word2_prob_vec) / \
                                    (np.linalg.norm(word1_prob_vec) * np.linalg.norm(word2_prob_vec))

            self.gt_list.append(gt_similarity)
            self.pred_list.append(similarity_prediction)

            word1_cluster = self.text_model.underlying_model.predict_cluster(word1)[0]
            word2_cluster = self.text_model.underlying_model.predict_cluster(word2)[0]
            if word1_cluster == word2_cluster:
                self.same_cluster_count += 1
                self.same_cluster_sim_sum += gt_similarity
            else:
                self.diff_cluster_count += 1
                self.diff_cluster_sim_sum += gt_similarity

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return  # Nothing to do here, this metric uses an external dataset

    def calc_results(self):
        self.results = {}
        self.traverse_validation_set()

        gt_and_predictions = np.array([self.gt_list, self.pred_list])
        pred_pearson_corr = np.corrcoef(gt_and_predictions)[0, 1]
        self.results['word ' + self.name + ' pearson correlation'] = pred_pearson_corr

        self.results['same_cluster_count'] = self.same_cluster_count
        self.results['same_cluster_mean_sim'] = self.same_cluster_sim_sum/self.same_cluster_count
        self.results['diff_cluster_count'] = self.diff_cluster_count
        self.results['diff_cluster_mean_sim'] = self.diff_cluster_sim_sum / self.diff_cluster_count

    def report(self):
        if self.results is None:
            self.calc_results()

        res = ''
        res += 'word ' + self.name + ' pearson correlation: ' + \
               self.precision_str % self.results['word ' + self.name + ' pearson correlation']

        res += ', same cluster: count ' + str(self.results['same_cluster_count']) + ', '
        res += 'mean similarity: ' + self.precision_str % self.results['same_cluster_mean_sim'] + ', '
        res += ', diff cluster: count ' + str(self.results['diff_cluster_count']) + ', '
        res += 'mean similarity: ' + self.precision_str % self.results['diff_cluster_mean_sim']

        return res

    @staticmethod
    def uses_external_dataset():
        return True

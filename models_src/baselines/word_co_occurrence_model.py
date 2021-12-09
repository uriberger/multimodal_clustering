import numpy as np
from sklearn.cluster import KMeans
from dataset_builders.concreteness_dataset import generate_concreteness_dataset


class WordCoOccurrenceModel:

    def __init__(self):
        self.word_to_ind = {}
        self.ind_to_word = []
        self.cur_ind = 0

        self.co_occurrence_matrix = None

    def document_word(self, word):
        if word not in self.word_to_ind:
            self.word_to_ind[word] = self.cur_ind
            self.ind_to_word.append(word)
            self.cur_ind += 1

    def document_word_occurrence(self, word):
        word_ind = self.word_to_ind[word]

        self.co_occurrence_matrix[word_ind, word_ind] += 1

    def document_co_occurrence(self, word1, word2):
        word1_ind = self.word_to_ind[word1]
        word2_ind = self.word_to_ind[word2]

        self.co_occurrence_matrix[word1_ind, word2_ind] += 1
        self.co_occurrence_matrix[word2_ind, word1_ind] += 1

    def create_co_occurrence_matrix(self):
        word_num = len(self.word_to_ind)
        self.co_occurrence_matrix = np.zeros((word_num, word_num))

    def predict_word_concreteness(self, token_count):
        concreteness_dataset = generate_concreteness_dataset()
        norm_co_oc_mat = (self.co_occurrence_matrix.transpose() /
                          np.linalg.norm(self.co_occurrence_matrix, axis=1)).transpose()

        common_words_concreteness = [x for x in concreteness_dataset.items()
                                     if x[0] in token_count and token_count[x[0]] > 10]
        common_words_concreteness.sort(key=lambda x: x[1])
        repr_word_per_type_num = 20
        concrete_words = [x[0] for x in common_words_concreteness[-repr_word_per_type_num:]]
        non_concrete_words = [x[0] for x in common_words_concreteness[:repr_word_per_type_num]]

        concrete_word_inds = [self.word_to_ind[w] for w in concrete_words]
        concrete_mat = self.co_occurrence_matrix[concrete_word_inds]
        non_concrete_word_inds = [self.word_to_ind[w] for w in non_concrete_words]
        non_concrete_mat = self.co_occurrence_matrix[non_concrete_word_inds]

        concreteness_mat = np.concatenate([concrete_mat, (-1)*non_concrete_mat])
        norm_conc_mat = (concreteness_mat.transpose() / np.linalg.norm(concreteness_mat, axis=1)).transpose()

        sim_with_repr_words = np.matmul(norm_co_oc_mat, norm_conc_mat.transpose())
        concreteness_prediction = np.sum(sim_with_repr_words, axis=1)

        return {
            self.ind_to_word[i]: concreteness_prediction[i]
            for i in range(len(self.ind_to_word))
        }

    def categorize_words(self, cluster_num):
        normalized_mat = (self.co_occurrence_matrix/np.linalg.norm(self.co_occurrence_matrix, axis=0)).transpose()
        # normalized_mat = (self.co_occurrence_matrix / np.sum(self.co_occurrence_matrix, axis=0)).transpose()
        kmeans = KMeans(n_clusters=cluster_num).fit(normalized_mat)
        cluster_list = list(kmeans.labels_)
        return {
            self.ind_to_word[i]: cluster_list[i]
            for i in range(len(self.ind_to_word))
        }

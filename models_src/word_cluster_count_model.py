import numpy as np


class WordClusterCountModel:

    def __init__(self, cluster_num, mode):
        self.cluster_num = cluster_num
        if mode == 'discriminative':
            self.word_to_cluster_co_occur = {}
            self.mode = 0
        elif mode == 'generative':
            self.cluster_to_word_co_occur = []
            for _ in range(cluster_num):
                self.cluster_to_word_co_occur.append({})
            self.cluster_to_word_prob = []
            self.cluster_prob = []
            self.overall_occur_num = 0
            self.mode = 1
        else:
            print('Error: no such mode ' + str(mode))
            assert False

    def document_co_occurrence(self, word, cluster_ind):
        if self.mode == 0:
            if word not in self.word_to_cluster_co_occur:
                self.word_to_cluster_co_occur[word] = [0] * self.cluster_num
            self.word_to_cluster_co_occur[word][cluster_ind] += 1
        else:
            if word not in self.cluster_to_word_co_occur[cluster_ind]:
                self.cluster_to_word_co_occur[cluster_ind][word] = 0
            self.cluster_to_word_co_occur[cluster_ind][word] += 1
            self.overall_occur_num += 1

    def calculate_probs(self):
        if self.mode == 0:
            return
        else:
            self.cluster_to_word_prob = []
            cluster_count = []
            for cluster_ind in range(self.cluster_num):
                word_count_dic = self.cluster_to_word_co_occur[cluster_ind]
                total_occurrence_num = sum(word_count_dic.values())
                cluster_count.append(total_occurrence_num)
                self.cluster_to_word_prob.append({
                    x[0]: x[1] / total_occurrence_num for x in word_count_dic.items()
                })

            # self.cluster_prob = [cluster_count[i]/self.overall_occur_num
            #                             for i in range(self.cluster_num)]
            # Use uniform class distribution
            self.cluster_prob = [1 / self.cluster_num for _ in range(self.cluster_num)]

    def get_cluster_conditioned_on_word(self, word):
        cluster_to_prob = [self.cluster_to_word_prob[x][word]
                           if word in self.cluster_to_word_prob[x]
                           else 0
                           for x in range(self.cluster_num)]

        # word_occur_count = sum(
        #     self.cluster_to_word_co_occur[x][word]
        #     if word in self.cluster_to_word_co_occur[x]
        #     else 0
        #     for x in range(self.cluster_num)
        # )
        # if word_occur_count == 0:
        #     # print('Never encountered word \'' + word + '\'.')
        #     return None
        # word_prob = word_occur_count/self.overall_occur_num
        #
        # # Apply Bayes rule
        # p_class_cond_word = [(class_to_prob[i]*self.cluster_prob[i])/word_prob
        #                      for i in range(self.cluster_num)]
        word_prob_sum = sum(cluster_to_prob)
        if word_prob_sum == 0:
            # print('Never encountered word \'' + word + '\'.')
            return None

        p_cluster_cond_word = [cluster_to_prob[x] / word_prob_sum for x in range(self.cluster_num)]
        return p_cluster_cond_word

    def predict_cluster(self, word):
        if self.mode == 0:
            if word not in self.word_to_cluster_co_occur:
                # print('Never encountered word \'' + word + '\'.')
                return None

            highest_correlated_cluster = np.argmax(self.word_to_cluster_co_occur[word])
            highest_count = self.word_to_cluster_co_occur[word][highest_correlated_cluster]
            overall_count = sum(self.word_to_cluster_co_occur[word])
            probability = highest_count / overall_count
            most_probable_class = highest_correlated_cluster
        else:
            p_class_cond_word = self.get_cluster_conditioned_on_word(word)
            if p_class_cond_word is None:
                return None
            probability = max(p_class_cond_word)
            most_probable_class = np.argmax(p_class_cond_word)
        return most_probable_class, probability

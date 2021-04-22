import numpy as np


class NounIdentifier:

    def __init__(self, semantic_class_num, mode):
        self.semantic_class_num = semantic_class_num
        if mode == 'discriminative':
            self.word_to_semantic_class_co_occur = {}
            self.mode = 0
        elif mode == 'generative':
            self.semantic_class_to_word_co_occur = []
            for _ in range(semantic_class_num):
                self.semantic_class_to_word_co_occur.append({})
            self.semantic_class_to_word_prob = []
            self.semantic_class_prob = []
            self.overall_occur_num = 0
            self.mode = 1
        else:
            print('Error: no such mode ' + str(mode))
            assert False

    def document_co_occurrence(self, word, semantic_class_ind):
        if self.mode == 0:
            if word not in self.word_to_semantic_class_co_occur:
                self.word_to_semantic_class_co_occur[word] = [0] * self.semantic_class_num
            self.word_to_semantic_class_co_occur[word][semantic_class_ind] += 1
        else:
            if word not in self.semantic_class_to_word_co_occur[semantic_class_ind]:
                self.semantic_class_to_word_co_occur[semantic_class_ind][word] = 0
            self.semantic_class_to_word_co_occur[semantic_class_ind][word] += 1
            self.overall_occur_num += 1

    def calculate_probs(self):
        if self.mode == 0:
            return
        else:
            self.semantic_class_to_word_prob = []
            semantic_class_count = []
            for semantic_class_ind in range(self.semantic_class_num):
                word_count_dic = self.semantic_class_to_word_co_occur[semantic_class_ind]
                total_occurrence_num = sum(word_count_dic.values())
                semantic_class_count.append(total_occurrence_num)
                self.semantic_class_to_word_prob.append({
                    x[0]: x[1]/total_occurrence_num for x in word_count_dic.items()
                })

            # self.semantic_class_prob = [semantic_class_count[i]/self.overall_occur_num
            #                             for i in range(self.semantic_class_num)]
            # Use uniform class distribution
            self.semantic_class_prob = [1/self.semantic_class_num for _ in range(self.semantic_class_num)]

    def predict_semantic_class(self, word):
        if self.mode == 0:
            if word not in self.word_to_semantic_class_co_occur:
                # print('Never encountered word \'' + word + '\'.')
                return None

            highest_correlated_semantic_class = np.argmax(self.word_to_semantic_class_co_occur[word])
            highest_count = self.word_to_semantic_class_co_occur[word][highest_correlated_semantic_class]
            overall_count = sum(self.word_to_semantic_class_co_occur[word])
            probability = highest_count / overall_count
            most_probable_class = highest_correlated_semantic_class
        else:
            class_to_prob = [self.semantic_class_to_word_prob[x][word]
                             if word in self.semantic_class_to_word_prob[x]
                             else 0
                             for x in range(self.semantic_class_num)]

            # word_occur_count = sum(
            #     self.semantic_class_to_word_co_occur[x][word]
            #     if word in self.semantic_class_to_word_co_occur[x]
            #     else 0
            #     for x in range(self.semantic_class_num)
            # )
            # if word_occur_count == 0:
            #     # print('Never encountered word \'' + word + '\'.')
            #     return None
            # word_prob = word_occur_count/self.overall_occur_num
            #
            # # Apply Bayes rule
            # p_class_cond_word = [(class_to_prob[i]*self.semantic_class_prob[i])/word_prob
            #                      for i in range(self.semantic_class_num)]
            word_prob_sum = sum(class_to_prob)
            if word_prob_sum == 0:
                # print('Never encountered word \'' + word + '\'.')
                return None

            p_class_cond_word = [class_to_prob[x]/word_prob_sum for x in range(self.semantic_class_num)]

            probability = max(p_class_cond_word)
            most_probable_class = np.argmax(p_class_cond_word)
        return most_probable_class, probability


def preprocess_token(token):
    token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!"))
    token = token.lower()

    return token

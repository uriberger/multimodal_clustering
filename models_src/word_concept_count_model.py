import numpy as np


class WordConceptCountModel:

    def __init__(self, concept_num, mode):
        self.concept_num = concept_num
        if mode == 'discriminative':
            self.word_to_concept_co_occur = {}
            self.mode = 0
        elif mode == 'generative':
            self.concept_to_word_co_occur = []
            for _ in range(concept_num):
                self.concept_to_word_co_occur.append({})
            self.concept_to_word_prob = []
            self.concept_prob = []
            self.overall_occur_num = 0
            self.mode = 1
        else:
            print('Error: no such mode ' + str(mode))
            assert False

    def document_co_occurrence(self, word, concept_ind):
        if self.mode == 0:
            if word not in self.word_to_concept_co_occur:
                self.word_to_concept_co_occur[word] = [0] * self.concept_num
            self.word_to_concept_co_occur[word][concept_ind] += 1
        else:
            if word not in self.concept_to_word_co_occur[concept_ind]:
                self.concept_to_word_co_occur[concept_ind][word] = 0
            self.concept_to_word_co_occur[concept_ind][word] += 1
            self.overall_occur_num += 1

    def calculate_probs(self):
        if self.mode == 0:
            return
        else:
            self.concept_to_word_prob = []
            concept_count = []
            for concept_ind in range(self.concept_num):
                word_count_dic = self.concept_to_word_co_occur[concept_ind]
                total_occurrence_num = sum(word_count_dic.values())
                concept_count.append(total_occurrence_num)
                self.concept_to_word_prob.append({
                    x[0]: x[1] / total_occurrence_num for x in word_count_dic.items()
                })

            # self.concept_prob = [concept_count[i]/self.overall_occur_num
            #                             for i in range(self.concept_num)]
            # Use uniform class distribution
            self.concept_prob = [1 / self.concept_num for _ in range(self.concept_num)]

    def predict_concept(self, word):
        if self.mode == 0:
            if word not in self.word_to_concept_co_occur:
                # print('Never encountered word \'' + word + '\'.')
                return None

            highest_correlated_concept = np.argmax(self.word_to_concept_co_occur[word])
            highest_count = self.word_to_concept_co_occur[word][highest_correlated_concept]
            overall_count = sum(self.word_to_concept_co_occur[word])
            probability = highest_count / overall_count
            most_probable_class = highest_correlated_concept
        else:
            concept_to_prob = [self.concept_to_word_prob[x][word]
                               if word in self.concept_to_word_prob[x]
                               else 0
                               for x in range(self.concept_num)]

            # word_occur_count = sum(
            #     self.concept_to_word_co_occur[x][word]
            #     if word in self.concept_to_word_co_occur[x]
            #     else 0
            #     for x in range(self.concept_num)
            # )
            # if word_occur_count == 0:
            #     # print('Never encountered word \'' + word + '\'.')
            #     return None
            # word_prob = word_occur_count/self.overall_occur_num
            #
            # # Apply Bayes rule
            # p_class_cond_word = [(class_to_prob[i]*self.concept_prob[i])/word_prob
            #                      for i in range(self.concept_num)]
            word_prob_sum = sum(concept_to_prob)
            if word_prob_sum == 0:
                # print('Never encountered word \'' + word + '\'.')
                return None

            p_class_cond_word = [concept_to_prob[x] / word_prob_sum for x in range(self.concept_num)]

            probability = max(p_class_cond_word)
            most_probable_class = np.argmax(p_class_cond_word)
        return most_probable_class, probability

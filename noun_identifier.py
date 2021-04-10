import numpy as np

class NounIdentifier:

    def __init__(self, semantic_class_num):
        self.semantic_class_num = semantic_class_num
        self.word_to_semantic_class_co_occur = {}

    def document_co_occurence(self, word, semantic_class_ind):
        if word not in self.word_to_semantic_class_co_occur:
            self.word_to_semantic_class_co_occur[word] = [0] * self.semantic_class_num
        self.word_to_semantic_class_co_occur[word][semantic_class_ind] += 1

    def predict_semantic_class(self, word):
        if word not in self.word_to_semantic_class_co_occur:
            # print('Never encoutered word \'' + word + '\'.')
            return None

        highest_correlated_semantic_class = np.argmax(self.word_to_semantic_class_co_occur[word])
        highest_count = self.word_to_semantic_class_co_occur[word][highest_correlated_semantic_class]
        overall_count = sum(self.word_to_semantic_class_co_occur[word])
        probability = highest_count / overall_count

        return highest_correlated_semantic_class, probability

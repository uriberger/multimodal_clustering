from loggable_object import LoggableObject
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import POS_LIST
import numpy as np
from sklearn import svm
from dataset_builders.fast_text import generate_fast_text


class ConcretenessSupervisedBaseline(LoggableObject):

    def __init__(self, use_pos_tags, use_suffix, use_embeddings, indent):
        super(ConcretenessSupervisedBaseline, self).__init__(indent)

        self.suffix_num = 200
        self.fast_text_dim = 300

        self.emb_dim = 0
        if use_pos_tags:
            self.emb_dim += len(POS_LIST)
        if use_suffix:
            self.emb_dim += self.suffix_num
        if use_embeddings:
            self.emb_dim += self.fast_text_dim

        self.use_pos_tags = use_pos_tags
        self.use_suffix = use_suffix
        self.use_embeddings = use_embeddings

    def get_pos_vector(self, word):
        res_vec = [len(wn.synsets(word, pos=pos_tag)) for pos_tag in POS_LIST]
        synset_num = sum(res_vec)
        if synset_num > 0:
            res_vec = [x/synset_num for x in res_vec]
        else:
            res_vec = [0]*len(POS_LIST)

        return np.array(res_vec)

    def find_common_suffixes(self, training_set):
        suffix_count = {}
        suffix_max_len = 4
        for word in training_set.keys():
            for suffix_len in range(1, suffix_max_len + 1):
                cur_suffix = word[-suffix_len:]
                if cur_suffix not in suffix_count:
                    suffix_count[cur_suffix] = 0
                suffix_count[cur_suffix] += 1

        all_suffixes = list(suffix_count.items())
        all_suffixes.sort(key=lambda x: x[1], reverse=True)
        self.common_suffixes = [x[0] for x in all_suffixes[:self.suffix_num]]

    def get_suffix_vector(self, word):
        res_vec = [
            1 if word.endswith(self.common_suffixes[i]) else 0
            for i in range(self.suffix_num)
        ]
        return np.array(res_vec)

    def get_embedding_vector(self, word):
        if word in self.fast_text:
            return self.fast_text[word]
        else:
            return np.zeros(self.fast_text_dim)

    def get_word_vector(self, word):
        word_vector = np.array([])
        if self.use_pos_tags:
            word_vector = np.concatenate([word_vector, self.get_pos_vector(word)])
        if self.use_suffix:
            word_vector = np.concatenate([word_vector, self.get_suffix_vector(word)])
        if self.use_embeddings:
            word_vector = np.concatenate([word_vector, self.get_embedding_vector(word)])

        return word_vector

    def train(self, training_set):
        self.log_print('Collecting common suffixes...')

        if self.use_suffix:
            self.find_common_suffixes(training_set)
        if self.use_embeddings:
            self.fast_text = generate_fast_text({x: True for x in training_set.keys()})

        X = np.zeros((len(training_set), self.emb_dim))
        y = np.zeros(len(training_set))
        word_to_ind = {}
        ind_to_word = []
        self.log_print('Collecting word vectors for all words...')
        for word, concreteness in training_set.items():
            cur_ind = len(ind_to_word)
            word_vector = self.get_word_vector(word)
            X[cur_ind, :] = word_vector
            y[cur_ind] = concreteness

            word_to_ind[word] = cur_ind
            ind_to_word.append(word)

        self.log_print('Training SVM...')
        regr = svm.SVR()
        regr.fit(X, y)

        self.log_print('Predicting using trained model...')
        predictions = regr.predict(X)

        return {
            ind_to_word[i]: predictions[i] for i in range(len(ind_to_word))
        }

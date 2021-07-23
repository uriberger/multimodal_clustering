import torch
import torch.nn as nn
import torchvision.models as models
from executors.trainer import Trainer
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import generate_textual_model
import os
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np
from utils.general_utils import for_loop_with_reports

vectors_filename = 'word2vec_cache'


def generate_all_word_vectors():
    ''' Generate the word vectors of the top frequent 500,000 words, from the GoogleNews data set.
    If there's a cache file- load it. Otherwise, generate from scrach, and save it to the cache
    file. '''
    fname = get_tmpfile(vectors_filename)
    if os.path.exists(fname):
        word_vectors = KeyedVectors.load(fname, mmap='r')
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            'word2vec-google-news-300.gz', binary=True, limit=500000
        )

    return model


class ClusterComparator(Trainer):

    def __init__(self, image_set, indent, model_name):
        super().__init__(image_set, 1, 100, indent)

        visual_model_dir = os.path.join(self.models_dir, 'visual')
        textual_model_dir = os.path.join(self.models_dir, 'text')
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, indent + 1, model_name)
        self.visual_model.eval()
        self.text_model = generate_textual_model(self.device, 'counts_generative', textual_model_dir, indent + 1,
                                                 model_name)

        self.image_embedder = models.resnet18(pretrained=True).to(self.device)
        self.image_embedder.fc = nn.Identity()
        image_embedding_dim = 512
        image_num = len(image_set)

        text_embedder = generate_all_word_vectors()
        text_embedding_dim = 300
        list_of_list_of_words = [list(x.keys()) for x in self.text_model.model.concept_to_word_co_occur]
        all_words = [word for word_list in list_of_list_of_words for word in word_list]
        vocab = list(set(all_words))
        word_num = len(vocab)
        concept_num = self.visual_model.config.concept_num

        self.image_embedding_mat = torch.zeros(image_num, image_embedding_dim).to(self.device)
        self.image_concept_mat = torch.zeros(image_num, concept_num).to(self.device)

        self.text_embedding_mat = np.zeros((word_num, text_embedding_dim))
        self.text_concept_mat = torch.zeros(word_num, concept_num).to(self.device)
        word_ind = 0
        for word in vocab:
            if word in text_embedder.key_to_index:
                word_index_in_word2vec = text_embedder.key_to_index[word]
            else:
                word_index_in_word2vec = 0
            self.text_embedding_mat[word_ind, :] = text_embedder.vectors[word_index_in_word2vec, :]
            self.text_concept_mat[word_ind, :] = torch.tensor(self.text_model.predict_concepts_for_word(word)).to(self.device)
            word_ind += 1

        self.index_to_image_id = {}

    def dump_results(self):
        torch.save([self.image_cluster_list, self.image_concept_mat, self.text_cluster_list, self.text_concept_mat],
                   'cluster_results')

    def pre_training(self):
        return

    def outer_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting outer index ' + str(index) +
                       ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))

    def outer_pair_loop(self, first_index, first_cluster, print_info):
        for second_index in range(first_index, len(self.cluster_list)):
            second_cluster = self.cluster_list[second_index]
            shared_concept_num = self.shared_concept_num_mat[first_index, second_index]

            if first_cluster == second_cluster:
                # Only according to image, these images are on the same cluster
                if shared_concept_num == 0:
                    self.only_im_sim_with_text_diff.append((first_index, second_index))
            else:
                if shared_concept_num > 3:
                    self.only_im_diff_with_text_sim.append((first_index, second_index))

    def post_training(self):
        visual_kmeans = KMeans(n_clusters=65).fit(self.image_embedding_mat.detach().numpy())
        textual_kmeans = KMeans(n_clusters=65).fit(self.text_embedding_mat)
        self.image_cluster_list = list(visual_kmeans.labels_)
        self.text_cluster_list = list(textual_kmeans.labels_)
        self.dump_results()

        self.log_print('Image:')
        self.shared_concept_num_mat = torch.matmul(self.image_concept_mat,
                                                   torch.transpose(self.image_concept_mat, 1, 0))
        self.cluster_list = self.image_cluster_list
        self.increment_indent()
        self.pairs_diff()
        self.decrement_indent()

        self.log_print('Text:')
        self.shared_concept_num_mat = torch.matmul(self.text_concept_mat,
                                                   torch.transpose(self.text_concept_mat, 1, 0))
        self.cluster_list = self.text_cluster_list
        self.increment_indent()
        self.pairs_diff()
        self.decrement_indent()

    def pairs_diff(self):
        # Go over all sample pairs and search for differences between clusters and concepts
        self.only_im_sim_with_text_diff = []
        self.only_im_diff_with_text_sim = []

        checkpoint_len = 100
        self.increment_indent()
        for_loop_with_reports(self.cluster_list, len(self.cluster_list), checkpoint_len,
                              self.outer_pair_loop, self.outer_progress_report)
        self.decrement_indent()

        self.log_print('Number of pairs similar when clustering by image but different when adding text: '
                       + str(len(self.only_im_sim_with_text_diff)))
        self.log_print('Number of pairs different when clustering by image but similar when adding text: '
                       + str(len(self.only_im_diff_with_text_sim)))

        only_im_sim_with_text_diff = [(self.index_to_image_id[x[0]], self.index_to_image_id[x[1]])
                                      for x in self.only_im_sim_with_text_diff]
        only_im_diff_with_text_sim = [(self.index_to_image_id[x[0]], self.index_to_image_id[x[1]])
                                      for x in self.only_im_diff_with_text_sim]
        self.log_print('List of pairs similar when clustering by image but different when adding text: '
                       + str(only_im_sim_with_text_diff[:10]))
        self.log_print('List of pairs different when clustering by image but similar when adding text: '
                       + str(only_im_diff_with_text_sim))

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        image_id = sampled_batch['image_id']
        image_indices = sampled_batch['index']

        # Infer
        self.visual_model.inference(image_tensor)

        labels_by_visual = self.visual_model.predict_concept_indicators()

        image_embedding = self.image_embedder(image_tensor)
        self.image_embedding_mat[image_indices, :] = image_embedding

        self.image_concept_mat[image_indices, :] = labels_by_visual
        for i in range(image_indices.shape[0]):
            self.index_to_image_id[image_indices[i].item()] = image_id[i].item()

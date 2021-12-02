from models_src.model_wrapper import ModelWrapper
from utils.visual_utils import generate_visual_model
from utils.text_utils import generate_text_model
import torch
import torch.nn as nn
import pytorch_metric_learning.losses as losses
import pytorch_metric_learning.distances as distances


class MultimodalClusterClassifier(nn.Module):

    def __init__(self, config):
        super(MultimodalClusterClassifier, self).__init__()
        self.visual_model = generate_visual_model(config.visual_model, config.cluster_num,
                                                  config.pretrained_visual_base_model)
        self.text_model = generate_text_model(config.text_model, config.cluster_num, config.word_embed_dim)

    def forward(self, visual_input, text_input):
        visual_output = self.visual_model(visual_input)

        text_output = self.text_model(text_input)[0]
        ''' The current 'text_output' variable contains the hidden states of each word in the sequence, for each
            sentence in the batch. However, the expected output of the model is a value for each cluster and each
            sentence, representing how likely it is that this cluster is instantiated in this sentence. So, we'll take
            the maximum of the values from all the words as the proxy for this probability. '''
        text_output = torch.max(text_output, dim=1).values

        return visual_output, text_output


class MultimodalModelWrapper(ModelWrapper):

    def __init__(self, device, config, model_dir, indent):
        """ The init function is used both for creating a new instance (when config is specified), and for loading
        saved instances (when config is None). """
        super(MultimodalModelWrapper, self).__init__(device, config, model_dir, 'multimodal', indent)

        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = losses.ContrastiveLoss(distance=distances.CosineSimilarity(),
                                                pos_margin=1, neg_margin=0)
        self.model.to(self.device)

        learning_rate = self.config.visual_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.cached_output = None

        # We need to keep a dictionary that will tell us the index of each word in the embedding matrix
        self.word_to_idx = {}
        self.word_to_idx[''] = 0

    def generate_model(self):
        return MultimodalClusterClassifier(self.config)

    def training_step(self):
        # loss = self.criterion(self.cached_output[0], self.cached_output[1])
        """ The contrastive lost calculated is over a matrix of embeddings (each row is an embedding of one sample)
        where labels defines positive/negative samples: each label is a class, positive pairs of samples are pairs of
        embeddings from the same class, negative pairs of samples are samples of embeddings from different classes.
        In our use case, we have batch_size embeddings created from the visual model (cached_output[0]) and
        batch_size embeddings created from the textual model (cached_output[1]). Each visual embedding corresponds
        to the textual embedding with the same index, and only to it. So, for the embedding mat we stack the visual
        embeddings over the textual embeddings, and then we'll have 2*batch_size rows, where the following are positive
        pairs: (0, batch_size), (1, batch_size+1), ... and all other are negative. """
        batch_size = self.cached_output[0].shape[0]
        embeddings = torch.cat((self.cached_output[0], self.cached_output[1]))
        labels = torch.cat((torch.tensor(range(batch_size)), torch.tensor(range(batch_size))))

        # Create positive and negative lists
        # Positive
        pos_first = torch.tensor(range(2*batch_size))
        pos_second = torch.cat((torch.tensor(range(batch_size, 2*batch_size)), torch.tensor(range(batch_size))))

        # Negative
        neg_first_lists = [[x]*(batch_size-1) for x in range(2*batch_size)]
        neg_first_unified_list = [inner for outer in neg_first_lists for inner in outer]
        neg_first = torch.tensor(neg_first_unified_list)
        neg_second_first_half_lists = [[x for x in range(batch_size, 2 * batch_size) if x != y + batch_size]
                                       for y in range(batch_size)]
        neg_second_second_half_lists = [[x for x in range(batch_size) if x != y - batch_size]
                                        for y in range(batch_size, 2 * batch_size)]
        neg_second_lists = neg_second_first_half_lists + neg_second_second_half_lists
        neg_second_unified_list = [inner for outer in neg_second_lists for inner in outer]
        neg_second = torch.tensor(neg_second_unified_list)

        indices_tuple = (pos_first, pos_second, neg_first, neg_second)

        loss = self.criterion(embeddings, labels, indices_tuple)
        loss_val = loss.item()
        self.cached_loss = loss_val

        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()

    def inference(self, visual_inputs, text_inputs):
        # Preprocess text input
        list_of_sets = [set(sentence) for sentence in text_inputs]
        all_words = set().union(*list_of_sets)
        for word in all_words:
            if word not in self.word_to_idx:  # Never seen this word before, document it
                self.word_to_idx[word] = len(self.word_to_idx)

        indices_of_inputs = [[self.word_to_idx[x] for x in sentence] for sentence in text_inputs]

        input_lengths = [len(sentence) for sentence in text_inputs]
        longest_sent_len = max(input_lengths)
        batch_size = len(text_inputs)

        # create an empty matrix with padding tokens
        pad_token_ind = self.word_to_idx['']
        text_input_tensor = torch.ones(batch_size, longest_sent_len, dtype=torch.long) * pad_token_ind
        text_input_tensor = text_input_tensor.to(self.device)

        # copy over the actual sequences
        for i, sent_len in enumerate(input_lengths):
            sequence = indices_of_inputs[i]
            text_input_tensor[i, 0:sent_len] = torch.tensor(sequence[:sent_len])

        visual_output, textual_output = self.model(visual_inputs, text_input_tensor)
        self.cached_output = (visual_output, textual_output)
        return visual_output, textual_output

    def print_info_on_inference(self):
        visual_clusters_indicator, text_clusters_indicator = self.predict_multimodal_cluster_indicators()

        res = 'Predicted ' + str(torch.sum(visual_clusters_indicator).item()) + ' clusters for image and '
        res += str(torch.sum(text_clusters_indicator).item()) + ' clusters for text '

        intersection_size = torch.sum(visual_clusters_indicator * text_clusters_indicator).item()
        res += 'out of which ' + str(intersection_size) + ' are shared'

        return res

    def print_info_on_loss(self):
        return 'Loss: ' + str(self.cached_loss)

    def dump_model(self):
        torch.save(self.model.state_dict(), self.get_model_path())

    def load_model(self):
        self.model.load_state_dict(torch.load(self.get_model_path(), map_location=torch.device(self.device)))

    def predict_multimodal_cluster_indicators(self):
        visual_output, text_output = self.cached_output

        with torch.no_grad():
            # Visual cluster indicators
            prob_output = torch.sigmoid(visual_output)
            visual_clusters_indicator = torch.zeros(prob_output.shape).to(self.device)
            visual_clusters_indicator[prob_output > self.config.object_threshold] = 1

            # Textual cluster indicators
            prob_output = torch.sigmoid(text_output)
            text_clusters_indicator = torch.zeros(prob_output.shape).to(self.device)
            text_clusters_indicator[prob_output > self.config.noun_threshold] = 1

        return visual_clusters_indicator, text_clusters_indicator

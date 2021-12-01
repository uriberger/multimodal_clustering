from models_src.unimodal_model_wrapper import UnimodalModelWrapper
from models_src.baselines.word_co_occurrence_model import WordCoOccurrenceModel
import torch
import numpy as np
from models_src.word_concept_count_model import WordConceptCountModel
import abc
from utils.text_utils import generate_text_model
from utils.text_utils import prepare_data


def generate_textual_model(device, config_or_str, dir_name, model_name, indent):
    """ Generate a textual model.
    Config can either be a model configuration, in case of a new model,
    or a string representing the name of the model, in case we want to
    load an existing model. """
    count_models = ['counts_generative', 'counts_discriminative']
    rnn_models = ['lstm', 'gru']

    if isinstance(config_or_str, str):
        config = None
        model_str = config_or_str
    else:
        config = config_or_str
        model_str = config_or_str.text_model
    if model_str in count_models:
        return TextualCountsModelWrapper(device, config, dir_name, model_name, indent)
    elif model_str in rnn_models:
        return TextualRNNModelWrapper(device, config, dir_name, model_name, indent)
    else:
        return None


def generate_textual_counts_model(model_str, concept_num):
    if model_str == 'counts_generative':
        model = WordConceptCountModel(concept_num, 'generative')
    elif model_str == 'counts_discriminative':
        model = WordConceptCountModel(concept_num, 'discriminative')

    return model


class TextualModelWrapper(UnimodalModelWrapper):

    def __init__(self, device, config, model_dir, model_name, indent):
        super(TextualModelWrapper, self).__init__(device, config, model_dir, model_name, indent)

    @abc.abstractmethod
    def predict_concept_insantiating_words(self, sentences):
        return

    @abc.abstractmethod
    def predict_concepts_for_word(self, word):
        return

    @abc.abstractmethod
    def predict_concept_for_word(self, word):
        return

    @abc.abstractmethod
    def estimate_concept_instantiation_per_word(self, sentences):
        return

    def print_info_on_inference(self):
        predictions_num = torch.sum(self.predict_concept_indicators()).item()
        return 'Predicted ' + str(predictions_num) + ' concepts according to text'


class TextualCountsModelWrapper(TextualModelWrapper):

    def __init__(self, device, config, model_dir, model_name, indent):
        super().__init__(device, config, model_dir, model_name, indent)
        self.model.calculate_probs()

    def generate_model(self):
        return generate_textual_counts_model(self.config.text_model, self.config.concept_num)

    def training_step(self, inputs, labels):
        loss = self.criterion(self.cached_output, labels)
        loss_val = loss.item()
        self.cached_loss = loss_val

        batch_size = len(inputs)

        for caption_ind in range(batch_size):
            predicted_concepts_by_image = [x for x in range(self.config.concept_num)
                                           if labels[caption_ind, x] == 1]
            char_list = inputs[caption_ind]
            for token_len in range(1, self.config.max_token_len + 1):
                token_list = [char_list[i:i+token_len] for i in range(len(char_list) - token_len + 1)]
                for token in token_list:
                    for concept_ind in predicted_concepts_by_image:
                        self.model.document_co_occurrence(token, concept_ind)

    def inference(self, inputs):
        self.model.calculate_probs()
        batch_size = len(inputs)

        with torch.no_grad():
            output_tensor = torch.zeros(batch_size, self.config.concept_num).to(self.device)
            for caption_ind in range(batch_size):
                cur_token = ''
                for char in inputs[caption_ind]:
                    cur_token += char
                    prediction_res = self.model.predict_concept(cur_token)
                    if prediction_res is None:
                        # Never seen this token before
                        continue
                    predicted_concept, prob = prediction_res
                    if prob >= self.config.noun_threshold:
                        # Found a token
                        if output_tensor[caption_ind, predicted_concept] < prob:
                            output_tensor[caption_ind, predicted_concept] = prob
                        cur_token = ''

        self.cached_output = output_tensor
        return output_tensor

    def dump_model(self):
        torch.save(self.model, self.get_model_path())

    def load_model(self):
        self.model = torch.load(self.get_model_path())

    def predict_concept_indicators(self):
        concept_indicators = torch.zeros(self.cached_output.shape).to(self.device)
        concept_indicators[self.cached_output > self.config.noun_threshold] = 1
        return concept_indicators

    def predict_concept_insantiating_words(self, sentences):
        res = []
        for sentence in sentences:
            res.append([])
            for token in sentence:
                concept_instantiating_token = False
                while True:
                    prediction_res = self.model.predict_concept(token)
                    if prediction_res is None:
                        # Never seen this token before
                        break
                    predicted_concept, prob = prediction_res
                    if prob >= self.config.noun_threshold:
                        concept_instantiating_token = True
                    break
                if concept_instantiating_token:
                    res[-1].append(1)
                else:
                    res[-1].append(0)

        return res

    def predict_concepts_for_word(self, word):
        concept_conditioned_on_word = self.model.get_concept_conditioned_on_word(word)
        if concept_conditioned_on_word is None:
            # return [0]*self.config.concept_num
            return None

        concept_indicators = [1 if x >= self.config.noun_threshold else 0 for x in concept_conditioned_on_word]
        return concept_indicators

    def predict_concept_for_word(self, word):
        concept_conditioned_on_word = self.model.get_concept_conditioned_on_word(word)
        if concept_conditioned_on_word is None:
            # return [0]*self.config.concept_num
            return None

        return np.argmax(concept_conditioned_on_word)

    def estimate_concept_instantiation_per_word(self, sentences):
        res = []
        for sentence in sentences:
            res.append([])
            for token in sentence:
                prediction_res = self.model.predict_concept(token)
                if prediction_res is None:
                    # Never seen this token before
                    res[-1].append(0)
                else:
                    predicted_concept, prob = prediction_res
                    res[-1].append(prob)

        return res

    def create_concept_to_gt_class_mapping(self, class_mapping):
        gt_class_to_prediction = {i: self.model.predict_concept(class_mapping[i])
                                  for i in class_mapping.keys()
                                  if ' ' not in class_mapping[i]}
        gt_class_to_concept = {x[0]: x[1][0] for x in gt_class_to_prediction.items() if x[1] is not None}
        concept_num = self.config.concept_num
        concept_to_gt_class = {concept_ind: [] for concept_ind in range(concept_num)}
        for gt_class_ind, concept_ind in gt_class_to_concept.items():
            concept_to_gt_class[concept_ind].append(gt_class_ind)

        return concept_to_gt_class


class TextualRNNModelWrapper(TextualModelWrapper):

    def __init__(self, device, config, model_dir, model_name, indent):
        super().__init__(device, config, model_dir, model_name, indent)
        self.model.to(self.device)

        learning_rate = self.config.textual_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # We need to keep a dictionary that will tell us the index of each word in the embedding matrix
        self.word_to_idx = {}
        self.word_to_idx[''] = 0

    def generate_model(self):
        model_str = self.config.text_model
        concept_num = self.config.concept_num
        word_embed_dim = self.config.word_embed_dim

        return generate_text_model(model_str, concept_num, word_embed_dim)

    def training_step(self, inputs, labels):
        loss = self.criterion(self.cached_output, labels)
        loss_val = loss.item()
        self.cached_loss = loss_val

        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()

    def inference(self, inputs):
        list_of_sets = [set(sentence) for sentence in inputs]
        all_words = set().union(*list_of_sets)
        for word in all_words:
            if word not in self.word_to_idx:  # Never seen this word before, document it
                self.word_to_idx[word] = len(self.word_to_idx)

        indices_of_inputs = [[self.word_to_idx[x] for x in sentence] for sentence in inputs]

        input_lengths = [len(sentence) for sentence in inputs]
        longest_sent_len = max(input_lengths)
        batch_size = len(inputs)

        # create an empty matrix with padding tokens
        pad_token_ind = self.word_to_idx['']
        input_tensor = torch.ones(batch_size, longest_sent_len, dtype=torch.long) * pad_token_ind
        input_tensor = input_tensor.to(self.device)

        # copy over the actual sequences
        for i, sent_len in enumerate(input_lengths):
            sequence = indices_of_inputs[i]
            input_tensor[i, 0:sent_len] = torch.tensor(sequence[:sent_len])

        output = self.model(input_tensor)[0]
        ''' The current 'output' variable contains the hidden states of each word in the sequence, for each
        sentence in the batch. However, the expected output of the model is a value for each concept and each
        sentence, representing how likely it is that this concept is instantiated in this sentence. So, we'll take
        the maximum of the values from all the words as the proxy for this probability. '''
        output = torch.max(output, dim=1).values
        self.cached_output = output
        return output

    def predict_concept_indicators(self):
        with torch.no_grad():
            prob_output = torch.sigmoid(self.cached_output)
            concepts_indicator = torch.zeros(prob_output.shape).to(self.device)
            concepts_indicator[prob_output > self.config.noun_threshold] = 1

        return concepts_indicator

    def dump_model(self):
        torch.save(self.model.state_dict(), self.get_model_path())

    def load_model(self):
        self.model.load_state_dict(torch.load(self.get_model_path(), map_location=torch.device(self.device)))

    def predict_concept_insantiating_words(self, sentences):
        res = []
        for sent_ind in range(len(sentences)):
            sentence = sentences[sent_ind]
            res.append([])
            for token_ind in range(len(sentence)):
                concept_instantiating_token = \
                    torch.max(self.cached_output[sent_ind, token_ind, :]) >= self.config.noun_threshold
                if concept_instantiating_token:
                    res[-1].append(1)
                else:
                    res[-1].append(0)

        return res

    def predict_concepts_for_word(self, word):
        old_cached_output = self.cached_output

        self.inference([[word]])
        concept_indicators = self.cached_output[0, 0, :] >= self.config.noun_threshold

        self.cached_loss = old_cached_output
        return concept_indicators


class TextualSimpleCountsModelWrapper(TextualModelWrapper):

    def __init__(self, device, config, model_dir, model_name, indent):
        super(TextualSimpleCountsModelWrapper, self).__init__(device, config, model_dir, model_name, indent)

    def generate_model(self):
        return WordCoOccurrenceModel()

    def predict_concept_insantiating_words(self, sentences):
        return

    def predict_concepts_for_word(self, word):
        return

    def predict_concept_for_word(self, word):
        if word in self.word_to_cluster:
            return self.word_to_cluster[word]
        else:
            return None

    def estimate_concept_instantiation_per_word(self, sentences):
        return

    def train(self, training_set):
        for sample in training_set.caption_data:
            caption = sample['caption']
            token_list = prepare_data([caption])[0]
            for token in token_list:
                self.model.document_word(token)

        self.model.create_co_occurrence_matrix()

        for sample in training_set.caption_data:
            caption = sample['caption']
            token_list = prepare_data([caption])[0]
            for token in token_list:
                self.model.document_word_occurrence(token)
            for i in range(len(token_list)):
                for j in range(i+1, len(token_list)):
                    self.model.document_co_occurrence(token_list[i], token_list[j])

        self.word_to_cluster = self.model.categorize_words(self.config.concept_num)

    def dump_model(self):
        torch.save([self.model, self.word_to_cluster], self.get_model_path())

    def load_model(self):
        self.model, self.word_to_cluster = torch.load(self.get_model_path())

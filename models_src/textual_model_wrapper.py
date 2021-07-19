from models_src.model_wrapper import ModelWrapper
import torch
import torch.nn as nn
from models_src.word_concept_count_model import WordConceptCountModel
import abc


def generate_textual_model(device, config, dir_name, indent):
    count_models = ['counts_generative', 'counts_discriminative']
    rnn_models = ['lstm', 'gru']

    if config.text_model in count_models:
        return TextualCountsModelWrapper(device, config, dir_name, indent)
    elif config.text_model in rnn_models:
        return TextualRNNModelWrapper(device, config, dir_name, indent)
    else:
        return None


def generate_textual_counts_model(model_str, concept_num):
    if model_str == 'counts_generative':
        model = WordConceptCountModel(concept_num, 'generative')
    elif model_str == 'counts_discriminative':
        model = WordConceptCountModel(concept_num, 'discriminative')

    return model


class TextualModelWrapper(ModelWrapper):

    def __init__(self, device, config, model_dir, indent, name=None):
        if name is None:
            name = 'textual'

        super(TextualModelWrapper, self).__init__(device, config, model_dir, indent, name)

    @abc.abstractmethod
    def predict_concept_insantiating_words(self, sentences):
        return

    def print_info_on_inference(self):
        predictions_num = torch.sum(self.predict_concept_indicators()).item()
        return 'Predicted ' + str(predictions_num) + ' concepts according to text'


class TextualCountsModelWrapper(TextualModelWrapper):

    def __init__(self, device, config, model_dir, indent, name=None):
        super().__init__(device, config, model_dir, indent, name)
        self.model.calculate_probs()

    def generate_model(self):
        return generate_textual_counts_model(self.config.text_model, self.config.concept_num)

    def training_step(self, inputs, labels):
        batch_size = len(inputs)

        for caption_ind in range(batch_size):
            predicted_concepts_by_image = [x for x in range(self.config.concept_num)
                                           if labels[caption_ind, x] == 1]
            for token in inputs[caption_ind]:
                for concept_ind in predicted_concepts_by_image:
                    self.model.document_co_occurrence(token, concept_ind)

    def inference(self, inputs):
        self.model.calculate_probs()
        batch_size = len(inputs)

        with torch.no_grad():
            output_tensor = torch.zeros(batch_size, self.config.concept_num).to(self.device)
            for caption_ind in range(batch_size):
                predicted_concept_list = []
                for token in inputs[caption_ind]:
                    prediction_res = self.model.predict_concept(token)
                    if prediction_res is None:
                        # Never seen this token before
                        continue
                    predicted_concept, prob = prediction_res
                    if prob >= self.config.noun_threshold:
                        predicted_concept_list.append(predicted_concept)
                output_tensor[caption_ind, torch.tensor(predicted_concept_list).long()] = 1.0

        self.cached_output = output_tensor
        return output_tensor

    def dump_model(self):
        torch.save(self.model, self.get_model_path())

    def load_model(self):
        self.model = torch.load(self.get_model_path())

    def predict_concept_indicators(self):
        return self.cached_output

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


class TextualRNNModelWrapper(TextualModelWrapper):

    def __init__(self, device, config, model_dir, indent, name=None):
        if name is None:
            name = 'visual'

        super().__init__(device, config, model_dir, indent, name)
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        learning_rate = self.config.textual_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.cached_loss = None

        # We need to keep a dictionary that will tell us the index of each word in the embedding matrix
        self.word_to_idx = {}
        self.word_to_idx[''] = 0

    def generate_model(self):
        model_str = self.config.text_model
        concept_num = self.config.concept_num
        word_embed_dim = self.config.word_embed_dim

        # We don't know what the size of the vocabulary will be, so let's take some large value
        vocab_size = 50000
        num_layers = 2

        if model_str == 'lstm':
            model = nn.Sequential(
                nn.Embedding(vocab_size, word_embed_dim),
                nn.LSTM(word_embed_dim, concept_num, num_layers, batch_first=True)
            )
        elif model_str == 'gru':
            model = nn.Sequential(
                nn.Embedding(vocab_size, word_embed_dim),
                nn.GRU(word_embed_dim, concept_num, num_layers, batch_first=True)
            )

        return model

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

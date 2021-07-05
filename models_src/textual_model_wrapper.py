from models_src.model_wrapper import ModelWrapper
import torch
from models_src.word_concept_count_model import WordConceptCountModel
import abc


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
    def predict_concept_insantiating_words(self, sentence):
        return


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

    def print_info_on_inference(self):
        predictions_num = torch.sum(self.cached_output).item()
        return 'Predicted ' + str(predictions_num) + ' concepts according to text'

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

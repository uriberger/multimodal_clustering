from executors.visual_evaluators.evaluate_visual_model import VisualModelEvaluator
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import TextualCountsModelWrapper
import clip
import torch
import abc
from metrics import VisualKnownClassesClassificationMetric


def clip_similarity_func(image_features, text_features):
    norm_image_features = image_features/image_features.norm(dim=-1, keepdim=True)
    norm_text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    return norm_image_features @ norm_text_features.T


def concept_similarity_func(image_concepts, text_concepts):
    hamming_dist = torch.sum(torch.fmod(image_concepts + text_concepts, 2))
    concept_num = image_concepts.shape[0]
    return concept_num - hamming_dist


class PromptEvaluator(VisualModelEvaluator):
    """ Evaluate a multi-modal model, that embeds images and text, by classifying
    using the method described in Radford et al.: given an image and a set of possible
    class names c1, c2, ..., cn, embed the image and the following n sentences: 'an
    image of a ci' for i in {1,...,n}, and classify the image to be the class with which
    the dot product is the highest. """
    def __init__(self, test_set, class_mapping, model_type, model_str, indent):
        super(PromptEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)
        self.metric = VisualKnownClassesClassificationMetric(None, len(class_mapping))

    def predict_visual_concepts_from_input(self, inputs):
        self.model.inference(inputs)
        return self.model.predict_concept_indicators()

    def predict_text_concepts_from_input(self, inputs):
        self.text_model.inference(inputs)
        return self.text_model.predict_concept_indicators()

    def clip_text_inference(self, inputs):
        model_inputs = clip.tokenize(inputs)
        return self.model.encode_text(model_inputs)

    def generate_model(self, model_type, model_str):
        if model_type == 'clip':
            model, _ = clip.load(model_str, self.device)
            inference_func = model.encode_image
            self.text_inference_func = self.clip_text_inference
            self.im_txt_similarity_func = clip_similarity_func
        elif model_type == 'unimodal':
            visual_model_wrapper = VisualModelWrapper(self.device, None, 'models/visual', self.indent + 1, model_str)
            text_model_wrapper = TextualCountsModelWrapper(self.device, None, 'models/text', self.indent + 1, model_str)

            model = visual_model_wrapper
            inference_func = self.predict_visual_concepts_from_input
            # inference_func = model.inference

            self.text_model = text_model_wrapper
            self.text_inference_func = self.predict_text_concepts_from_input
            # self.text_inference_func = text_model_wrapper.inference
            self.im_txt_similarity_func = concept_similarity_func
            # self.im_txt_similarity_func = clip_similarity_func
        else:
            # No such model type
            assert False
        return model, inference_func

    def metric_pre_calculations(self):
        prompts = {x: 'a photo of a ' + self.class_mapping[x] for x in self.class_mapping.keys()}
        # prompts = {x: self.class_mapping[x] for x in self.class_mapping.keys()}
        self.class_ind_to_embedding = {x: self.text_inference_func(prompts[x])
                                       for x in self.class_mapping.keys()}

    @abc.abstractmethod
    def predict_classes(self, sample_ind):
        return
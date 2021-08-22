import torch
import torch.nn as nn
from utils.visual_utils import generate_visual_model
from executors.visual_evaluators.evaluate_visual_model import VisualModelEvaluator
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.simclr import SimCLRModel, clean_state_dict
import clip
import abc
import os
from metrics import VisualUnknownClassesClassificationMetric


class ClusteringEvaluator(VisualModelEvaluator):
    """ Evaluate an embedding pre-trained model for self-supervised classification using clustering. """

    def __init__(self, test_set, class_mapping, model_type, model_str, indent):
        super(ClusteringEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)
        self.metric = VisualUnknownClassesClassificationMetric(None)

    def generate_model(self, model_type, model_str):
        if model_type == 'pretrained':
            model = generate_visual_model(model_str, 1, True)
            model.fc = nn.Identity()
            inference_func = model.forward
        elif model_type == 'clip':
            model, _ = clip.load(model_str, self.device)
            inference_func = model.encode_image
        elif model_type == 'unimodal':
            model_wrapper = VisualModelWrapper(self.device, None, 'models/visual', self.indent + 1, model_str)
            model = model_wrapper.model
            model.fc = nn.Identity()
            inference_func = model.forward
        elif model_type == 'simclr':
            ''' We have some issue of unnormalized weights for the simclr model, which causes it to be very slow.
            To solve this, we need the following line: '''
            torch.set_flush_denormal(True)
            model = SimCLRModel()
            model_path = os.path.join(VisualModelEvaluator.root_dir, model_str)
            model.load_state_dict(clean_state_dict(torch.load(model_path, map_location=self.device)))
            inference_func = lambda x: model.forward(x)[0]
        else:
            # No such model type
            assert False
        return model, inference_func

    @abc.abstractmethod
    def predict_classes(self, sample_ind):
        return

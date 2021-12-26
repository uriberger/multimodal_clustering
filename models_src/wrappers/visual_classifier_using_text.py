###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.


import abc
import clip
import torch
from loggable_object import LoggableObject


class VisualClassifierUsingText(LoggableObject):
    """ This class wraps a Vision and Language model, and classifies visual input to the ground-truth classes using the
        name of the classes. """

    def __init__(self, class_mapping, indent):
        super(VisualClassifierUsingText, self).__init__(indent)
        self.class_mapping = class_mapping

    """ Execute actions needed for initialization. """

    @abc.abstractmethod
    def initialize(self):
        return

    """ Run inference using the visual part of the underlying mode on the given input. """

    @abc.abstractmethod
    def inference(self, visual_inputs):
        return

    """ Used previously inferred results, along with the text part of the underlying model, to predict
        classification. """

    @abc.abstractmethod
    def classify_using_inferred_results(self):
        return

    """ Classify multiple visual inputs. We first run inference using the visual part of the underlying mode on the
        input, and then use these results along with the text part to predict the classification.
    """

    def classify(self, visual_inputs):
        self.inference(visual_inputs)
        return self.classify_using_inferred_results()


class ClusterVisualClassifier(VisualClassifierUsingText):
    """ This class use our multimodal clustering model to classify visual inputs. """

    def __init__(self, visual_model_wrapper, text_model_wrapper, class_mapping, indent):
        super(ClusterVisualClassifier, self).__init__(class_mapping, indent)
        self.visual_model_wrapper = visual_model_wrapper
        self.text_model_wrapper = text_model_wrapper
        self.initialize()

    def initialize(self):
        self.cluster_to_gt_class = self.text_model_wrapper.create_cluster_to_gt_class_mapping(self.class_mapping)

    def inference(self, visual_inputs):
        self.visual_model_wrapper.inference(visual_inputs)
        self.cached_output = self.visual_model_wrapper.predict_cluster_lists()

    def classify_using_inferred_results(self):
        predicted_clusters = self.cached_output

        batch_size = len(predicted_clusters)
        predicted_classes = []

        for sample_ind in range(batch_size):
            sample_predicted_clusters = predicted_clusters[sample_ind]
            sample_predicted_class_lists = [self.cluster_to_gt_class[x] for x in sample_predicted_clusters]
            sample_predicted_classes = [inner for outer in sample_predicted_class_lists for inner in outer]
            predicted_classes.append(sample_predicted_classes)

        return predicted_classes


class CLIPVisualClassifier(VisualClassifierUsingText):
    """ This class use the CLIP model to classify visual inputs. CLIP was introduced in the paper
        "Learning Transferable Visual Models From Natural Language Supervision" By Radford et al.
    """

    def __init__(self, positive_threshold, class_mapping, indent):
        super(CLIPVisualClassifier, self).__init__(class_mapping, indent)

        self.positive_threshold = positive_threshold
        self.model, _ = clip.load('clipRN50', self.device)
        self.initialize()

    def initialize(self):
        prompts = {x: 'a photo of a ' + self.class_mapping[x] for x in self.class_mapping.keys()
                   if ' ' not in self.class_mapping[x]}
        class_ind_and_embedding = [(x[0], self.model.encode_text(clip.tokenize(x[1]).to(self.device)).float())
                                   for x in prompts.items()]
        self.class_embedding_mat = torch.cat([x[1] for x in class_ind_and_embedding])
        self.class_embedding_mat = self.class_embedding_mat / self.class_embedding_mat.norm(dim=-1, keepdim=True)
        self.mat_ind_to_class_ind = {i: class_ind_and_embedding[i][0] for i in range(len(class_ind_and_embedding))}

    def inference(self, visual_inputs):
        self.cached_output = self.model.encode_image(visual_inputs)

    def classify_using_inferred_results(self):
        image_features = self.cached_output

        batch_size = image_features.shape[0]
        class_num = image_features.shape[1]

        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity_mat = norm_image_features @ self.class_embedding_mat.T

        # Check which image-class similarities exceed the threshold
        similarity_mat[similarity_mat >= self.positive_threshold] = 1
        similarity_mat[similarity_mat < self.positive_threshold] = 0

        class_indicator_lists = [[1 if similarity_mat[sample_ind, mat_class_ind] >= self.positive_threshold else 0
                                  for mat_class_ind in range(class_num)] for sample_ind in range(batch_size)]

        return [[self.mat_ind_to_class_ind[mat_class_ind] for mat_class_ind in range(class_num)
                 if class_indicator_lists[sample_ind][mat_class_ind]] for sample_ind in range(batch_size)]

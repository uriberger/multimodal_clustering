import abc
import torch
from utils.visual_utils import calc_ious
from utils.text_utils import noun_tags
from models_src.model_config import wanted_image_size
import numpy as np


class Metric:
    """ This class represents a metric for evaluating the model.
    It is given the model and inputs, generates a prediction, compares to
    the ground-truth and reports the evaluation of the specific metric. """

    def __init__(self, visual_model, text_model):
        self.visual_model = visual_model
        self.text_model = text_model

    """ Predicts the output of the model for a specific input, compares
    to ground-truth, and documents the current evaluation. """
    @abc.abstractmethod
    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    """ Reports some aggregation of evaluation of all inputs. """
    @abc.abstractmethod
    def report(self):
        return


class SensitivitySpecificityMetric(Metric):

    def __init__(self, visual_model, text_model):
        super(SensitivitySpecificityMetric, self).__init__(visual_model, text_model)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def report_with_name(self, name):
        res = name + ': '
        res += 'tp ' + str(self.tp)
        res += ', fp ' + str(self.fp)
        res += ', fn ' + str(self.fn)
        res += ', tn ' + str(self.tn)

        if self.tp + self.fp == 0:
            precision = 0
        else:
            precision = self.tp / (self.tp + self.fp)
        res += ', precision: ' + str(precision)

        if self.tp + self.fn == 0:
            recall = 0
        else:
            recall = self.tp / (self.tp + self.fn)
        res += ', recall: ' + str(recall)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        res += ', F1: ' + str(f1)

        return res

    @abc.abstractmethod
    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    @abc.abstractmethod
    def report(self):
        return


class BBoxMetric(SensitivitySpecificityMetric):

    """ This metric predicts bounding boxes, compares the prediction to the
    ground-truth bounding boxes (using intersection-over-union), and reports
    the results. """

    def __init__(self, visual_model):
        super(BBoxMetric, self).__init__(visual_model, None)

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_bboxes = self.visual_model.predict_bboxes(visual_inputs)
        iou_threshold = self.visual_model.config.object_threshold

        gt_bboxes = visual_metadata['gt_bboxes']

        batch_size = len(gt_bboxes)
        for sample_ind in range(batch_size):
            gt_bbox_num = len(gt_bboxes[sample_ind])
            sample_gt_bboxes = torch.stack([torch.tensor(x) for x in gt_bboxes[sample_ind]])
            if len(predicted_bboxes[sample_ind]) > 0:
                sample_predicted_bboxes = torch.stack([torch.tensor(x) for x in predicted_bboxes[sample_ind]])
                ious = calc_ious(sample_gt_bboxes, sample_predicted_bboxes)

            tp = 0
            fp = 0
            identifed_gt_inds = {}
            for predicted_bbox_ind in range(len(predicted_bboxes[sample_ind])):
                for gt_bbox_ind in range(gt_bbox_num):
                    iou = ious[gt_bbox_ind, predicted_bbox_ind]
                    if iou >= iou_threshold:
                        tp += 1
                        identifed_gt_inds[gt_bbox_ind] = True
                        continue
                    fp += 1
            fn = gt_bbox_num - len(identifed_gt_inds)

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def report(self):
        return self.report_with_name('bbox results')


class NounIdentificationMetric(SensitivitySpecificityMetric):
    """ This metric uses the model to predict if a word is a noun (by asking
    if it instantiates a concept). It then compares the prediction to the
    ground-truth (extracted from a pretrained pos tagger) and reports the
    results. """

    def __init__(self, text_model, nlp):
        super(NounIdentificationMetric, self).__init__(None, text_model)
        self.nlp = nlp

    def prepare_ground_truth(self, text_inputs):
        gt_res = []
        batch_size = len(text_inputs)
        for sample_ind in range(batch_size):
            gt_res.append([])
            doc = self.nlp(' '.join(text_inputs[sample_ind]))
            for token in doc:
                is_noun_gt = token.tag_ in noun_tags
                if is_noun_gt:
                    gt_res[-1].append(1)
                else:
                    gt_res[-1].append(0)

        return gt_res

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predictions = self.text_model.predict_concept_insantiating_words(text_inputs)
        text_gt = self.prepare_ground_truth(text_inputs)
        for i in range(len(text_inputs)):
            is_noun_prediction = (predictions[i] == 1)
            is_noun_gt = (text_gt[i] == 1)
            if is_noun_prediction and is_noun_gt:
                self.tp += 1
            elif is_noun_prediction and (not is_noun_gt):
                self.fp += 1
            elif (not is_noun_prediction) and is_noun_gt:
                self.fn += 1
            else:
                self.tn += 1

    def report(self):
        return self.report_with_name('Noun prediction results')


class ConcretenessPredictionMetric(Metric):
    """ This metric uses a concreteness dataset: a human-annotated dataset
    that gives every word a number between 1 and 5, representing how
    concrete is this word. We compare it to our prediction: a word that
    instantiates a concept is concrete (5) and is otherwise in-concrete (1).
    """

    def __init__(self, text_model, concreteness_dataset):
        super(ConcretenessPredictionMetric, self).__init__(None, text_model)
        self.concreteness_dataset = concreteness_dataset
        self.absolute_error_sum = 0
        self.count = 0

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        concept_inst_predictions = self.text_model.predict_concept_insantiating_words(text_inputs)
        ''' Concreteness should be between 1 and 5. We have a number between
        0 and 1. So we scale it to the range [1, 5] '''
        concreteness_predictions = [[1 + 4*x for x in y] for y in concept_inst_predictions]

        batch_size = len(text_inputs)
        for sample_ind in range(batch_size):
            token_list = text_inputs[sample_ind]
            for i in range(len(token_list)):
                token = token_list[i]
                if token not in self.concreteness_dataset:
                    continue
                self.count += 1
                gt_concreteness = self.concreteness_dataset[token]
                predicted_concreteness = concreteness_predictions[sample_ind][i]
                self.absolute_error_sum += abs(gt_concreteness - predicted_concreteness)

    def report(self):
        mae = self.absolute_error_sum / self.count
        return 'Concreteness mean squared error: ' + str(mae)


class SentenceImageMatchingMetric(Metric):
    """ This metric chooses 2 random samples, and checks if the model knows
    to align the correct sentence to the correct image.
    This is performed by predicting the concepts for one image, and for the
    two sentences, and checking the hamming distance of the concepts vector
    predicted according to the image to that predicted according to each
    sentence. If the hamming distance is lower from the correct sentence,
    this is considered a correct prediction. """

    def __init__(self, visual_model, text_model):
        super(SentenceImageMatchingMetric, self).__init__(visual_model, text_model)
        self.correct_count = 0
        self.overall_count = 0

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        self.visual_model.inference(visual_inputs)
        concepts_by_image = self.visual_model.predict_concept_indicators()
        self.text_model.inference(text_inputs)
        concepts_by_text = self.text_model.predict_concept_indicators()

        batch_size = len(text_inputs) // 2
        for pair_sample_ind in range(batch_size):
            single_sample_ind = 2 * pair_sample_ind
            sample_concepts_by_image = concepts_by_image[single_sample_ind]
            sample_concepts_by_first_caption = concepts_by_text[single_sample_ind]
            sample_concepts_by_second_caption = concepts_by_text[single_sample_ind + 1]
            first_hamming_distance = torch.sum(
                torch.abs(
                    sample_concepts_by_image - sample_concepts_by_first_caption
                )
            ).item()
            second_hamming_distance = torch.sum(
                torch.abs(
                    sample_concepts_by_image - sample_concepts_by_second_caption
                )
            ).item()

            if first_hamming_distance < second_hamming_distance:
                self.correct_count += 1
            self.overall_count += 1

    def report(self):
        accuracy = self.correct_count / self.overall_count
        return 'Image sentence alignment accuracy: ' + str(accuracy)


class VisualClassificationMetric(SensitivitySpecificityMetric):
    """ Base class for image multi-label classification metric. """

    def __init__(self, visual_model):
        super(VisualClassificationMetric, self).__init__(visual_model, None)

    def evaluate_classification(self, predicted_classes, gt_classes):
        predicted_num = len(predicted_classes)
        cur_tp = len(list(set(predicted_classes).intersection(gt_classes)))
        cur_fp = predicted_num - cur_tp

        class_num = self.visual_model.config.concept_num
        non_predicted_num = class_num - predicted_num
        cur_fn = len(list(set(gt_classes).difference(predicted_classes)))
        cur_tn = non_predicted_num - cur_fn

        self.tp += cur_tp
        self.fp += cur_fp
        self.fn += cur_fn
        self.tn += cur_tn


class VisualKnownClassesClassificationMetric(VisualClassificationMetric):
    """ This metric is only for models trained with labels of classes.
    We train a visual model to predict the classes on an image, and evaluate its predictions, given the ground
    truth classes. """

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        self.visual_model.inference(visual_inputs)
        predicted_classes = self.visual_model.predict_concept_lists()

        gt_classes_str = visual_metadata['gt_classes']
        gt_classes = [int(x) for x in gt_classes_str.split(',')]

        self.evaluate_classification(predicted_classes, gt_classes)

    def report(self):
        return self.report_with_name('Visual classification results')


class VisualUnknownClassesClassificationMetric(VisualClassificationMetric):
    """ This metric is only for models that have labels of classes in their evaluation set, but were not trained
    with these labels.
    We train a visual model to cluster images, and evaluate its clustering by mapping the clusters to the
    labelled classes in a many-to-one manner, by maximizing the F1 score. """

    def __init__(self, visual_model):
        super(VisualUnknownClassesClassificationMetric, self).__init__(visual_model, None)

        # Maintain a list of pairs of (predicted clusters, gt classes) for future calculations
        self.predicted_clusters_gt_classes = []

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        self.visual_model.inference(visual_inputs)
        predicted_classes = self.visual_model.predict_concept_lists()
        self.predicted_clusters_gt_classes.append((predicted_classes, visual_metadata['gt_classes']))

    def evaluate(self):
        concept_num = self.visual_model.config.concept_num

        # First, document for each concept how many times each class co-occurred with it
        concept_class_co_occur = []
        for _ in range(concept_num):
            concept_class_co_occur.append({})

        for predicted_concepts, gt_classes in self.predicted_clusters_gt_classes:
            for predicted_concept in predicted_concepts:
                for gt_class in gt_classes:
                    if gt_class not in concept_class_co_occur[predicted_concept]:
                        concept_class_co_occur[predicted_concept][gt_class] = 0
                    concept_class_co_occur[predicted_concept][gt_class] += 1

        # Now, for each concept, choose the class with which it co-occurred the most
        concept_to_class = {
            x: max(concept_class_co_occur[x], key=concept_class_co_occur[x].get)
            for x in range(concept_num)
        }

        # Finally, go over the results again and use the mapping to evaluate
        for predicted_concepts, gt_classes in self.predicted_clusters_gt_classes:
            predicted_classes = [concept_to_class[x] for x in predicted_concepts]
            self.evaluate_classification(predicted_classes, gt_classes)

    def report(self):
        """In this metric we have post analysis, we'll do it in the report function as this function is
        executed after all calculations are done."""
        self.evaluate()
        return self.report_with_name('Visual classification results')

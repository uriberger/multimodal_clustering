import abc
import torch
from utils.visual_utils import calc_ious, get_resized_gt_bboxes
from utils.text_utils import noun_tags
import statistics
import numpy as np
from sklearn.metrics.cluster import v_measure_score


class Metric:
    """ This class represents a metric for evaluating the model.
    It is given the model and inputs, generates a prediction, compares to
    the ground-truth and reports the evaluation of the specific metric. """

    def __init__(self, visual_model, text_model):
        self.visual_model = visual_model
        self.text_model = text_model
        self.results = None
        self.precision_str = "%.4f"

    """ Predicts the output of the model for a specific input, compares
    to ground-truth, and documents the current evaluation. """

    @abc.abstractmethod
    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    """ Reports some aggregation of evaluation of all inputs. """

    @abc.abstractmethod
    def report(self):
        return

    """ Returns a mapping from metric name to metric value. """

    @abc.abstractmethod
    def calc_results(self):
        return

    """ A flag to indicate whether this metric is only related to images. """

    def is_image_only(self):
        return False

    """ A flag to indicate whether this metric uses an external dataset. """

    def uses_external_dataset(self):
        return False


class SensitivitySpecificityMetric(Metric):

    def __init__(self, visual_model, text_model):
        super(SensitivitySpecificityMetric, self).__init__(visual_model, text_model)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def calc_results(self):
        if self.results is None:
            self.results = {}
        name = self.get_name()

        if self.tp + self.fp == 0:
            precision = 0
        else:
            precision = self.tp / (self.tp + self.fp)
        self.results[name + ' precision'] = precision

        if self.tp + self.fn == 0:
            recall = 0
        else:
            recall = self.tp / (self.tp + self.fn)
        self.results[name + ' recall'] = recall

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        self.results[name + ' F1'] = f1

    def report_with_name(self):
        if self.results is None:
            self.calc_results()

        name = self.get_name()
        res = name + ': '
        res += 'tp ' + str(self.tp)
        res += ', fp ' + str(self.fp)
        res += ', fn ' + str(self.fn)
        res += ', tn ' + str(self.tn)

        res += ', precision: ' + self.precision_str % self.results[name + ' precision']
        res += ', recall: ' + self.precision_str % self.results[name + ' recall']
        res += ', F1: ' + self.precision_str % self.results[name + ' F1']

        return res

    @abc.abstractmethod
    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    @abc.abstractmethod
    def report(self):
        return

    @abc.abstractmethod
    def get_name(self):
        return


class BBoxPredictionMetric(SensitivitySpecificityMetric):
    """ This is a class from which the bbox and heatmap metrics will inherit. """

    def __init__(self, visual_model):
        super(BBoxPredictionMetric, self).__init__(visual_model, None)

    @abc.abstractmethod
    def map_pred_to_bbox(self, sample_predicted_heatmaps, sample_gt_bboxes):
        return

    @staticmethod
    def try_matching_pred_to_bbox(final_mapping, pred_ind, bbox_ind, matched_preds, matched_bboxes):
        if pred_ind not in matched_preds and bbox_ind not in matched_bboxes:
            final_mapping[pred_ind] = bbox_ind
            matched_preds[pred_ind] = True
            matched_bboxes[bbox_ind] = True

    @staticmethod
    def build_bbox_to_pred_map(pred_to_matching_bbox, matched_preds, matched_bboxes):
        bbox_to_matching_pred = {}

        for pred_ind, bbox_index_list in pred_to_matching_bbox.items():
            if pred_ind not in matched_preds:
                for bbox_ind in bbox_index_list:
                    if bbox_ind not in matched_bboxes:
                        if bbox_ind not in bbox_to_matching_pred:
                            bbox_to_matching_pred[bbox_ind] = []
                        bbox_to_matching_pred[bbox_ind].append(pred_ind)

        return bbox_to_matching_pred

    def find_best_pred_bbox_matching(self, pred_to_matching_bbox):
        """ We can have multiple prediction-bbox matching. We use a heuristic to find the best one:
        First, we match predictions/bboxes with only one mapping. Only then we match others. """
        final_mapping = {}
        matched_preds = {}
        matched_bboxes = {}
        # First search for heatmaps with a single matching bbox
        for pred_ind, bbox_index_list in pred_to_matching_bbox.items():
            if len(bbox_index_list) == 1:
                self.try_matching_pred_to_bbox(final_mapping, pred_ind, bbox_index_list[0],
                                               matched_preds, matched_bboxes)

        # Next, search for bboxes with a single matching heatmap
        bbox_to_matching_heatmap = self.build_bbox_to_pred_map(pred_to_matching_bbox,
                                                                  matched_preds, matched_bboxes)

        for bbox_ind, heatmap_index_list in bbox_to_matching_heatmap.items():
            if len(heatmap_index_list) == 1:
                self.try_matching_pred_to_bbox(final_mapping, heatmap_index_list[0], bbox_ind,
                                               matched_preds, matched_bboxes)

        # Finally, match unmatched
        bbox_to_matching_pred = self.build_bbox_to_pred_map(pred_to_matching_bbox,
                                                            matched_preds, matched_bboxes)
        for bbox_ind, pred_index_list in bbox_to_matching_pred.items():
            while len(pred_index_list) > 0:
                self.try_matching_pred_to_bbox(final_mapping, pred_index_list[0], bbox_ind,
                                                  matched_preds, matched_bboxes)
                pred_index_list = pred_index_list[1:]

        return final_mapping

    def document(self, orig_image_sizes, predicted_list, gt_bboxes):
        batch_size = len(gt_bboxes)
        for sample_ind in range(batch_size):
            sample_gt_bboxes = gt_bboxes[sample_ind]
            gt_bbox_num = len(sample_gt_bboxes)
            sample_gt_bboxes = get_resized_gt_bboxes(sample_gt_bboxes, orig_image_sizes[sample_ind])
            sample_predicted = predicted_list[sample_ind]
            predicted_num = len(sample_predicted)

            # Map predicted to matching bboxes
            pred_to_matching_bbox = self.map_pred_to_bbox(sample_predicted, sample_gt_bboxes)

            # Determine final prediction -> bbox mapping
            final_mapping = self.find_best_pred_bbox_matching(pred_to_matching_bbox)

            tp = len(final_mapping)
            fp = predicted_num - tp
            fn = gt_bbox_num - tp

            self.tp += tp
            self.fp += fp
            self.fn += fn


class BBoxMetric(BBoxPredictionMetric):
    """ This metric predicts bounding boxes, compares the prediction to the
    ground-truth bounding boxes (using intersection-over-union), and reports
    the results. """

    def __init__(self, visual_model):
        super(BBoxMetric, self).__init__(visual_model)

    def map_pred_to_bbox(self, sample_predicted_bboxes, sample_gt_bboxes):
        gt_bbox_num = len(sample_gt_bboxes)
        sample_gt_bboxes = torch.stack([torch.tensor(x) for x in sample_gt_bboxes])
        if len(sample_predicted_bboxes) > 0:
            sample_predicted_bboxes = torch.stack([torch.tensor(x) for x in sample_predicted_bboxes])
            ious = calc_ious(sample_gt_bboxes, sample_predicted_bboxes)
        pred_to_matching_gt = {}
        for predicted_bbox_ind in range(len(sample_predicted_bboxes)):
            pred_to_matching_gt[predicted_bbox_ind] = []
            for gt_bbox_ind in range(gt_bbox_num):
                if ious[gt_bbox_ind, predicted_bbox_ind] >= 0.5:
                    pred_to_matching_gt[predicted_bbox_ind].append(gt_bbox_ind)

        return pred_to_matching_gt

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_bboxes = self.visual_model.predict_bboxes(visual_inputs)
        gt_bboxes = visual_metadata['gt_bboxes']
        orig_image_sizes = visual_metadata['orig_image_size']
        self.document(orig_image_sizes, predicted_bboxes, gt_bboxes)

    def document_with_loaded_results(self, orig_image_sizes, activation_maps, gt_bboxes):
        predicted_bboxes = self.visual_model.predict_bboxes_with_activation_maps(activation_maps)
        self.document(orig_image_sizes, predicted_bboxes, gt_bboxes)

    def report(self):
        return self.report_with_name()

    def get_name(self):
        return 'bbox prediction'

    def is_image_only(self):
        return True


class HeatmapMetric(BBoxPredictionMetric):
    """ This metric measures whether the center of a predicted heatmap is inside the gt bounding box. """

    def __init__(self, visual_model):
        super(HeatmapMetric, self).__init__(visual_model, None)

    def map_pred_to_bbox(self, sample_predicted_heatmaps, sample_gt_bboxes):
        gt_bbox_num = len(sample_gt_bboxes)
        heatmap_to_matching_bbox = {}
        for predicted_heatmap_ind in range(len(sample_predicted_heatmaps)):
            heatmap_to_matching_bbox[predicted_heatmap_ind] = []
            predicted_heatmap = sample_predicted_heatmaps[predicted_heatmap_ind]
            max_heatmap_ind = torch.argmax(predicted_heatmap)
            max_heatmap_loc = (torch.div(max_heatmap_ind, predicted_heatmap.shape[1], rounding_mode='floor').item(),
                               max_heatmap_ind % predicted_heatmap.shape[1])

            for gt_bbox_ind in range(gt_bbox_num):
                gt_bbox = sample_gt_bboxes[gt_bbox_ind]
                if gt_bbox[0] <= max_heatmap_loc[0] <= gt_bbox[2] and \
                        gt_bbox[1] <= max_heatmap_loc[1] <= gt_bbox[3]:
                    heatmap_to_matching_bbox[predicted_heatmap_ind].append(gt_bbox_ind)

        return heatmap_to_matching_bbox

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_heatmaps = [self.visual_model.get_heatmaps_without_inference()]
        gt_bboxes = visual_metadata['gt_bboxes']
        orig_image_sizes = visual_metadata['orig_image_size']
        self.document(orig_image_sizes, predicted_heatmaps, gt_bboxes)

    def report(self):
        return self.report_with_name()

    def get_name(self):
        return 'heatmap prediction'

    def is_image_only(self):
        return True


class NounIdentificationMetric(SensitivitySpecificityMetric):
    """ This metric uses the model to predict if a word is a noun (by asking
    if it instantiates a cluster). It then compares the prediction to the
    ground-truth (extracted from a pretrained pos tagger) and reports the
    results. """

    def __init__(self, text_model, nlp):
        super(NounIdentificationMetric, self).__init__(None, text_model)
        self.nlp = nlp
        self.noun_count = 0
        self.non_noun_count = 0

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
                    self.noun_count += 1
                else:
                    gt_res[-1].append(0)
                    self.non_noun_count += 1

        return gt_res

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predictions = self.text_model.predict_cluster_insantiating_words(text_inputs)
        text_gt = self.prepare_ground_truth(text_inputs)

        batch_size = len(text_inputs)
        for sentence_ind in range(batch_size):
            sentence_len = len(text_inputs[sentence_ind])
            for i in range(sentence_len):
                is_noun_prediction = (predictions[sentence_ind][i] == 1)
                is_noun_gt = (text_gt[sentence_ind][i] == 1)
                if is_noun_prediction and is_noun_gt:
                    self.tp += 1
                elif is_noun_prediction and (not is_noun_gt):
                    self.fp += 1
                elif (not is_noun_prediction) and is_noun_gt:
                    self.fn += 1
                else:
                    self.tn += 1

    def calc_majority_baseline_results(self):
        # Majority baseline means we always predict noun or always predict non-noun
        if self.noun_count > self.non_noun_count:
            # Better to always predict noun
            accuracy = self.noun_count / (self.noun_count + self.non_noun_count)
        else:
            # Better to always predict non-noun
            accuracy = self.non_noun_count / (self.noun_count + self.non_noun_count)

        return accuracy

    def report(self):
        majority_basline_accuracy = self.calc_majority_baseline_results()
        return self.report_with_name() + \
               ', majority baseline accuracy: ' + self.precision_str % majority_basline_accuracy

    def get_name(self):
        return 'Noun prediction'


class ConcretenessPredictionMetric(Metric):
    """ This metric uses a concreteness dataset: a human-annotated dataset
    that gives every word a number between 1 and 5, representing how
    concrete is this word. We compare it to our prediction: a word that
    instantiates a cluster is concrete (5) and is otherwise in-concrete (1).
    """

    def __init__(self, text_model, concreteness_dataset, token_count, predicted_concreteness=None):
        super(ConcretenessPredictionMetric, self).__init__(None, text_model)
        self.concreteness_dataset = concreteness_dataset
        self.estimation_absolute_error_sum = 0
        self.estimation_absolute_error_sum_for_conc_words = 0
        self.estimation_absolute_error_sum_for_non_conc_words = 0
        self.prediction_absolute_error_sum = 0
        self.prediction_absolute_error_sum_for_conc_words = 0
        self.prediction_absolute_error_sum_for_non_conc_words = 0
        self.tested_words_count = 0
        self.conc_words_count = 0
        self.non_conc_words_count = 0

        ''' We want to evaluate concreteness on different sets of words: words that appeared more than once
        in the training set, words that appeared more than 5 times in the training set, etc. '''
        self.token_count = token_count

        self.predicted_concreteness = predicted_concreteness

        self.min_count_vals = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        self.min_count_gt_lists = []
        self.min_count_est_lists = []
        self.min_count_pred_lists = []
        for _ in self.min_count_vals:
            self.min_count_gt_lists.append([])
            self.min_count_est_lists.append([])
            self.min_count_pred_lists.append([])

    def traverse_vocab(self):
        for token, count in self.token_count.items():
            if token not in self.concreteness_dataset:
                continue

            self.tested_words_count += 1
            gt_concreteness = self.concreteness_dataset[token]
            ''' We try both estimations of concreteness (a number between 0 and 1) and predictions of concreteness
            (a binary result on the estimation, after applying a threshold). '''
            if self.predicted_concreteness:
                concreteness_estimation = self.predicted_concreteness[token]
            else:
                concreteness_estimation = self.text_model.estimate_cluster_instantiation_per_word([[token]])[0][0]
            concreteness_prediction = concreteness_estimation >= self.text_model.config.text_threshold
            estimation_absolute_error = abs(gt_concreteness - concreteness_estimation)
            prediction_absolute_error = abs(gt_concreteness - concreteness_prediction)
            self.estimation_absolute_error_sum += estimation_absolute_error
            self.prediction_absolute_error_sum += prediction_absolute_error

            if concreteness_prediction == 1:  # concrete
                self.conc_words_count += 1
                self.estimation_absolute_error_sum_for_conc_words += estimation_absolute_error
                self.prediction_absolute_error_sum_for_conc_words += prediction_absolute_error
            else:  # Non concrete
                self.non_conc_words_count += 1
                self.estimation_absolute_error_sum_for_non_conc_words += estimation_absolute_error
                self.prediction_absolute_error_sum_for_non_conc_words += prediction_absolute_error

            for i in range(len(self.min_count_vals)):
                val = self.min_count_vals[i]
                if count > val:
                    self.min_count_gt_lists[i].append(gt_concreteness)
                    self.min_count_est_lists[i].append(concreteness_estimation)
                    self.min_count_pred_lists[i].append(concreteness_prediction)

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    def calc_results(self):
        self.results = {}
        self.traverse_vocab()

        concreteness_estimation_mae = 0
        concreteness_prediction_mae = 0
        if self.tested_words_count > 0:
            concreteness_estimation_mae = \
                self.estimation_absolute_error_sum / self.tested_words_count
            concreteness_prediction_mae = \
                self.prediction_absolute_error_sum / self.tested_words_count
        self.results['concreteness estimation mae'] = concreteness_estimation_mae
        self.results['concreteness prediction mae'] = concreteness_prediction_mae

        concreteness_estimation_mae_for_conc = 0
        concreteness_prediction_mae_for_conc = 0
        if self.conc_words_count > 0:
            concreteness_estimation_mae_for_conc = \
                self.estimation_absolute_error_sum_for_conc_words / self.conc_words_count
            concreteness_prediction_mae_for_conc = \
                self.prediction_absolute_error_sum_for_conc_words / self.conc_words_count
        self.results['concreteness estimation mae for conc'] = concreteness_estimation_mae_for_conc
        self.results['concreteness prediction mae for conc'] = concreteness_prediction_mae_for_conc

        concreteness_estimation_mae_for_non_conc = 0
        concreteness_prediction_mae_for_non_conc = 0
        if self.non_conc_words_count > 0:
            concreteness_estimation_mae_for_non_conc = \
                self.estimation_absolute_error_sum_for_non_conc_words / self.non_conc_words_count
            concreteness_prediction_mae_for_non_conc = \
                self.prediction_absolute_error_sum_for_non_conc_words / self.non_conc_words_count
        self.results['concreteness estimation mae for non conc'] = concreteness_estimation_mae_for_non_conc
        self.results['concreteness prediction mae for non conc'] = concreteness_prediction_mae_for_non_conc

        for i in range(len(self.min_count_vals)):
            gt_and_estimations = np.array([self.min_count_gt_lists[i], self.min_count_est_lists[i]])
            gt_and_predictions = np.array([self.min_count_gt_lists[i], self.min_count_pred_lists[i]])
            est_pearson_corr = np.corrcoef(gt_and_estimations)[0, 1]
            pred_pearson_corr = np.corrcoef(gt_and_predictions)[0, 1]
            self.results['concreteness estimation/prediction correlation over ' + str(self.min_count_vals[i])] = \
                (est_pearson_corr, pred_pearson_corr, len(self.min_count_gt_lists[i]))

    def report(self):
        if self.results is None:
            self.calc_results()

        res = ''
        res += 'Concreteness estimation mean absolute error: ' + \
               self.precision_str % self.results['concreteness estimation mae'] + ' overall, '
        res += \
            self.precision_str % self.results['concreteness estimation mae for conc'] + \
            ' for concrete-predicted words, '
        res += \
            self.precision_str % self.results['concreteness estimation mae for non conc'] + \
            ' for non-concrete-predicted words, '
        res += 'concreteness prediction mean absolute error: ' + \
               self.precision_str % self.results['concreteness prediction mae'] + ' overall, '
        res += \
            self.precision_str % self.results['concreteness prediction mae for conc'] + \
            ' for concrete-predicted words, '
        res += \
            self.precision_str % self.results['concreteness prediction mae for non conc'] + \
            ' for non-concrete-predicted words, '

        res += 'pearson correlation by token count: '
        for val in self.min_count_vals:
            if val != self.min_count_vals[0]:
                res += ', '
            res += str(val) + ': '
            cur_result = self.results['concreteness estimation/prediction correlation over ' + str(val)]
            res += str((self.precision_str % cur_result[0], self.precision_str % cur_result[1], cur_result[2]))

        return res

    def uses_external_dataset(self):
        return True


class SentenceImageMatchingMetric(Metric):
    """ This metric chooses 2 random samples, and checks if the model knows
    to align the correct sentence to the correct image.
    This is performed by predicting the clusters for one image, and for the
    two sentences, and checking the hamming distance of the clusters vector
    predicted according to the image to that predicted according to each
    sentence. If the hamming distance is lower from the correct sentence,
    this is considered a correct prediction. """

    def __init__(self, visual_model, text_model):
        super(SentenceImageMatchingMetric, self).__init__(visual_model, text_model)
        self.correct_count = 0
        self.incorrect_count = 0
        self.overall_count = 0

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        clusters_by_image = self.visual_model.predict_cluster_indicators()
        clusters_by_text = self.text_model.predict_cluster_indicators()

        batch_size = len(text_inputs) // 2
        for pair_sample_ind in range(batch_size):
            single_sample_ind = 2 * pair_sample_ind
            sample_clusters_by_image = clusters_by_image[single_sample_ind]
            sample_clusters_by_first_caption = clusters_by_text[single_sample_ind]
            sample_clusters_by_second_caption = clusters_by_text[single_sample_ind + 1]
            first_hamming_distance = torch.sum(
                torch.abs(
                    sample_clusters_by_image - sample_clusters_by_first_caption
                )
            ).item()
            second_hamming_distance = torch.sum(
                torch.abs(
                    sample_clusters_by_image - sample_clusters_by_second_caption
                )
            ).item()

            if first_hamming_distance < second_hamming_distance:
                self.correct_count += 1
            if first_hamming_distance > second_hamming_distance:
                self.incorrect_count += 1
            self.overall_count += 1

    def calc_results(self):
        self.results = {
            'image sentence alignment accuracy': self.correct_count / self.overall_count,
            'image sentence alignment extended accuracy': 1 - (self.incorrect_count / self.overall_count)
        }

    def report(self):
        if self.results is None:
            self.calc_results()

        return 'Image sentence alignment accuracy: ' + \
               self.precision_str % self.results['image sentence alignment accuracy'] + ', ' + \
               'extended accuracy: ' + \
               self.precision_str % self.results['image sentence alignment extended accuracy']


class VisualClassificationMetric(SensitivitySpecificityMetric):
    """ Base class for image multi-label classification metric. """

    def __init__(self, visual_model):
        super(VisualClassificationMetric, self).__init__(visual_model, None)

    def evaluate_classification(self, predicted_classes, gt_classes):
        batch_size = len(predicted_classes)
        for sample_ind in range(batch_size):
            sample_predicted = predicted_classes[sample_ind]
            sample_gt = gt_classes[sample_ind]
            predicted_num = len(sample_predicted)
            cur_tp = len(list(set(sample_predicted).intersection(sample_gt)))
            cur_fp = predicted_num - cur_tp

            non_predicted_num = self.class_num - predicted_num
            cur_fn = len(list(set(sample_gt).difference(sample_predicted)))
            cur_tn = non_predicted_num - cur_fn

            self.tp += cur_tp
            self.fp += cur_fp
            self.fn += cur_fn
            self.tn += cur_tn

    @abc.abstractmethod
    def document(self, predicted_classes, gt_classes):
        return

    def is_image_only(self):
        return True


class VisualKnownClassesClassificationMetric(VisualClassificationMetric):
    """ This metric is only for models trained with labels of classes.
    We train a visual model to predict the classes on an image, and evaluate its predictions, given the ground
    truth classes. """

    def __init__(self, visual_model, class_num):
        super(VisualKnownClassesClassificationMetric, self).__init__(visual_model)
        self.class_num = class_num

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_classes = self.visual_model.predict_cluster_lists()

        gt_classes_str = visual_metadata['gt_classes']
        gt_classes = [int(x) for x in gt_classes_str.split(',')]
        self.document(predicted_classes, gt_classes)

    def document(self, predicted_classes, gt_classes):
        self.evaluate_classification(predicted_classes, gt_classes)

    def report(self):
        return self.report_with_name()

    def get_name(self):
        return 'Visual classification'


class VisualUnknownClassesClassificationMetric(VisualClassificationMetric):
    """ This metric is only for models that have labels of classes in their evaluation set, but were not trained
    with these labels.
    We train a visual model to cluster images, and evaluate its clustering by mapping the clusters to the
    labelled classes in a many-to-one manner, by maximizing the F1 score. """

    def __init__(self, visual_model, mapping_mode):
        super(VisualUnknownClassesClassificationMetric, self).__init__(visual_model)

        # Maintain a list of pairs of (predicted clusters, gt classes) for future calculations
        self.predicted_clusters_gt_classes = []
        ''' Mapping mode: Two possible heuristics for choosing cluster to class mapping:
        First, for each cluster choose the class with which it co-occurred the most.
        Second, for each cluster choose the class with which it has the largest IoU. '''
        self.mapping_mode = mapping_mode

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_classes = self.visual_model.predict_cluster_lists()
        self.document(predicted_classes, visual_metadata['gt_classes'])

    def document(self, predicted_classes, gt_classes):
        batch_size = len(gt_classes)
        self.predicted_clusters_gt_classes += \
            [(predicted_classes[i], gt_classes[i]) for i in range(batch_size)]

    def evaluate(self):
        # Get a unique list of clusters
        clusters_with_repetition = [x[0] for x in self.predicted_clusters_gt_classes]
        cluster_list = list(set([inner for outer in clusters_with_repetition for inner in outer]))
        self.class_num = len(cluster_list)

        # First, document for each cluster how many times each class co-occurred with it
        cluster_class_co_occur = {x: {} for x in cluster_list}

        predicted_cluster_count = {x: 0 for x in cluster_list}
        gt_class_count = {}
        for predicted_clusters, gt_classes in self.predicted_clusters_gt_classes:
            for predicted_cluster in predicted_clusters:
                # Increment count
                predicted_cluster_count[predicted_cluster] += 1
                # Go over gt classes
                for gt_class in gt_classes:
                    # Increment count
                    if gt_class not in gt_class_count:
                        gt_class_count[gt_class] = 0
                    gt_class_count[gt_class] += 1
                    # Document co-occurrence
                    if gt_class not in cluster_class_co_occur[predicted_cluster]:
                        cluster_class_co_occur[predicted_cluster][gt_class] = 0
                    cluster_class_co_occur[predicted_cluster][gt_class] += 1

        ''' Two possible heuristics for choosing cluster to class mapping:
        First, for each cluster choose the class with which it co-occurred the most.
        Second, for each cluster choose the class with which it has the largest IoU. '''
        # First option: for each cluster, choose the class with which it co-occurred the most
        if self.mapping_mode == 'co_occur':
            cluster_to_class = {
                x: max(cluster_class_co_occur[x], key=cluster_class_co_occur[x].get)
              if len(cluster_class_co_occur[x]) > 0 else None
                for x in cluster_list
            }

            # Finally, go over the results again and use the mapping to evaluate
            for predicted_clusters, gt_classes in self.predicted_clusters_gt_classes:
                predicted_classes = [cluster_to_class[x] for x in predicted_clusters]
                self.evaluate_classification([predicted_classes], [gt_classes])

            # Apart from the classification results, we want to measure the intersection of our classes and the gt classes
            intersections = {x: cluster_class_co_occur[x][cluster_to_class[x]] for x in cluster_list}
            unions = {x: predicted_cluster_count[x] +  # Cluster count
                         gt_class_count[cluster_to_class[x]] -  # Class count
                         intersections[x]  # Intersection count
                      for x in cluster_list}
            self.ious = {x: intersections[x] / unions[x] if unions[x] > 0 else 0
                         for x in cluster_list}

            cluster_to_class = {
                x: max(cluster_class_co_occur[x], key=cluster_class_co_occur[x].get)
                if len(cluster_class_co_occur[x]) > 0 else None
                for x in cluster_list
            }
        # Second option: for each cluster choose the class with which it has the largest IoU
        elif self.mapping_mode == 'iou':
            intersections = cluster_class_co_occur
            unions = {
                cluster_ind: {
                    class_ind:
                        predicted_cluster_count[cluster_ind] +  # Cluster count
                        gt_class_count[class_ind] -  # Class count
                        intersections[cluster_ind][class_ind]  # Intersection count
                        if class_ind in intersections[cluster_ind] else 0
                    for class_ind in gt_class_count.keys()
                }
                for cluster_ind in cluster_list
            }
            ious = {
                cluster_ind: {
                    class_ind:
                        intersections[cluster_ind][class_ind] / unions[cluster_ind][class_ind]
                        if unions[cluster_ind][class_ind] > 0 else 0
                    for class_ind in gt_class_count.keys()
                }
                for cluster_ind in cluster_list
            }

            # Now, for each cluster, choose the class with which it co-occurred the most
            cluster_to_class = {
                x: max(ious[x], key=ious[x].get)
                if len(ious[x]) > 0 else None
                for x in cluster_list
            }

            self.ious = {cluster_ind: ious[cluster_ind][cluster_to_class[cluster_ind]] for cluster_ind in cluster_list}

        # Finally, go over the results again and use the mapping to evaluate
        for predicted_clusters, gt_classes in self.predicted_clusters_gt_classes:
            predicted_classes = [cluster_to_class[x] for x in predicted_clusters]
            self.evaluate_classification([predicted_classes], [gt_classes])

        self.cluster_to_class = cluster_to_class

    def report(self):
        """In this metric we have post analysis, we'll do it in the report function as this function is
        executed after all calculations are done."""
        self.evaluate()

        res = self.report_with_name()
        res += ', iou max ' + str(max(self.ious.values())) + ' min ' + str(min(self.ious.values())) + \
               ' mean ' + str(statistics.mean(self.ious.values())) + ' median ' + \
               str(statistics.median(self.ious.values())) + '\n'

        res += 'Cluster class pairs: '
        for cluster_ind, iou in self.ious.items():
            class_ind = self.cluster_to_class[cluster_ind]
            res += '(' + str(cluster_ind) + ',' + str(class_ind) + ',' + "{:.2f}".format(iou) + ') '

        return res

    def calc_results(self):
        self.results = {}

        if 'created_cluster_num' not in self.results:
            self.results['created_cluster_num'] = len(self.ious)
        SensitivitySpecificityMetric.calc_results(self)

    def get_name(self):
        return 'Visual classification ' + str(self.mapping_mode) + ' mapping'


class VisualPromptClassificationMetric(VisualClassificationMetric):
    """ This metric is for models mapping both images and text to the same space.
    We prompt the model by giving it the name of the ground-truth classes, and this enables us to map the model's
    clusters to gt classes. """

    def __init__(self, visual_model, text_model, class_mapping):
        super(VisualPromptClassificationMetric, self).__init__(visual_model)
        self.text_model = text_model
        self.cluster_to_gt_class = self.text_model.create_cluster_to_gt_class_mapping(class_mapping)
        self.class_num = len(class_mapping)

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        batch_size = len(text_inputs)
        predicted_clusters = self.visual_model.predict_cluster_lists()
        for sample_ind in range(batch_size):
            predicted_class_lists = [self.cluster_to_gt_class[x] for x in predicted_clusters[sample_ind]]
            predicted_classes = [inner for outer in predicted_class_lists for inner in outer]

            gt_classes = visual_metadata['gt_classes'][sample_ind]
            self.document([predicted_classes], [gt_classes])

    def document(self, predicted_classes, gt_classes):
        self.evaluate_classification(predicted_classes, gt_classes)

    def report(self):
        return self.report_with_name()

    def get_name(self):
        return 'Prompt classification'


class CategorizationMetric(Metric):
    """ This metric estimate whether the textual model categorizes words according to some baseline (the category
    dataset).
    We use the V-measure-score for evaluation. """

    def __init__(self, text_model, category_dataset, ignore_unknown_words=True):
        super(CategorizationMetric, self).__init__(None, text_model)
        self.category_dataset = category_dataset
        self.ignore_unknown_words = ignore_unknown_words

        if self.ignore_unknown_words:
            self.name_prefix_str = 'ignore_unknown'
        else:
            self.name_prefix_str = 'include_unknown'

    @staticmethod
    def fuzzy_v_measure_score(gt_class_to_sample, cluster_to_sample):

        if len(cluster_to_sample) == 0:
            return 0, 0, 0

        def safe_log(num):
            if num == 0:
                return 0
            else:
                return np.log2(num)

        sample_to_gt_class_count = {}
        for gt_class, sample_list in gt_class_to_sample.items():
            for sample in sample_list:
                if sample not in sample_to_gt_class_count:
                    sample_to_gt_class_count[sample] = 0
                sample_to_gt_class_count[sample] += 1

        sample_to_mass = {x[0]: 1 / x[1] for x in sample_to_gt_class_count.items()}
        clustering_mass = sum([len(y) for y in cluster_to_sample.values()])

        all_gt_classes = list(gt_class_to_sample.keys())
        all_clusters = list(cluster_to_sample.keys())
        cluster_list_for_all_gt_classes = [[(x, y) for y in all_clusters] for x in all_gt_classes]
        all_gt_classes_and_clusters = [inner for outer in cluster_list_for_all_gt_classes for inner in outer]
        gt_class_to_cluster_mass = {x: {} for x in all_gt_classes}
        cluster_to_gt_class_mass = {x: {} for x in all_clusters}

        for gt_class, gt_sample_list in gt_class_to_sample.items():
            for cluster, cluster_sample_list in cluster_to_sample.items():
                intersection = list(set(gt_sample_list).intersection(cluster_sample_list))
                intersection_mass = sum([sample_to_mass[x] for x in intersection])
                gt_class_to_cluster_mass[gt_class][cluster] = intersection_mass
                cluster_to_gt_class_mass[cluster][gt_class] = intersection_mass

        gt_class_to_mass = {x[0]: sum(x[1].values()) for x in gt_class_to_cluster_mass.items()}
        cluster_to_mass = {x[0]: sum(x[1].values()) for x in cluster_to_gt_class_mass.items()}

        p_gt = {x[0]: x[1] / clustering_mass for x in gt_class_to_mass.items()}
        p_cluster = {x[0]: x[1] / clustering_mass for x in cluster_to_mass.items()}
        p_gt_cluster = {x: gt_class_to_cluster_mass[x[0]][x[1]] / clustering_mass for x in all_gt_classes_and_clusters}

        H_gt = (-1) * sum([x * safe_log(x) for x in p_gt.values()])
        H_clusters = (-1) * sum([x * safe_log(x) for x in p_cluster.values()])
        H_gt_given_clusters = (-1) * sum([p_gt_cluster[x] * safe_log(p_gt_cluster[x] / p_cluster[x[1]])
                                          if p_cluster[x[1]] > 0 else 0 for x in all_gt_classes_and_clusters])
        H_clusters_given_gt = (-1) * sum([p_gt_cluster[x] * safe_log(p_gt_cluster[x] / p_gt[x[0]])
                                          if p_gt[x[0]] > 0 else 0 for x in all_gt_classes_and_clusters])

        homogeneity = 1 - H_gt_given_clusters / H_gt
        completeness = 1 - H_clusters_given_gt / H_clusters

        if homogeneity + completeness == 0:
            score = 0
        else:
            score = 2 * (homogeneity * completeness) / (homogeneity + completeness)

        return homogeneity, completeness, score

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        # This metric is not related to the test set, all the measurements are done later
        return

    def calculate_scatter_metric(self):
        gt_labels = []
        predicted_labels = []
        category_index = 0
        for category, word_list in self.category_dataset.items():
            for word in word_list:
                # Prediction
                prediction = self.text_model.predict_cluster_for_word(word)
                if prediction is not None:
                    # The word is known
                    predicted_labels.append(prediction)
                elif not self.ignore_unknown_words:
                    ''' The word is unknown, but we were told not to ignore unknown words, so we'll label it by a new
                    cluster. '''
                    new_cluster_ind = self.text_model.config.cluster_num
                    predicted_labels.append(new_cluster_ind)

                if (prediction is not None) or (not self.ignore_unknown_words):
                    ''' Only in case that we appended something to the predicted_labels list, we need to append to the
                    gt_labels list '''
                    gt_labels.append(category_index)

            category_index += 1
        self.results['v_measure_score'] = v_measure_score(gt_labels, predicted_labels)
        self.results['purity'], self.results['collocation'], self.results['pu_co_f1'] = \
            self.calc_purity_collocation(gt_labels, predicted_labels)
        self.results['FScore'] = self.calc_fscore(gt_labels, predicted_labels)

    def calculate_fuzzy_scatter_metric(self):
        all_words = list(set([word for inner in self.category_dataset.values() for word in inner]))
        cluster_to_word_list = {}
        for word in all_words:
            prediction = self.text_model.predict_clusters_for_word(word)
            if prediction is None:
                continue
            predicted_clusters = [x for x in range(len(prediction)) if prediction[x] == 1]
            for cluster in predicted_clusters:
                if cluster not in cluster_to_word_list:
                    cluster_to_word_list[cluster] = []
                cluster_to_word_list[cluster].append(word)

        self.results['fuzzy_homogeneity'], self.results['fuzzy_completeness'], self.results['fuzzy_v_measure_score'] = \
            self.fuzzy_v_measure_score(self.category_dataset, cluster_to_word_list)

    @staticmethod
    def calc_purity_collocation(gt_labels, predicted_labels):
        N = len(gt_labels)
        if N == 0:
            return 0, 0, 0

        cluster_to_gt_intersection = {}
        gt_to_cluster_intersection = {}
        for i in range(N):
            gt_class = gt_labels[i]
            predicted_cluster = predicted_labels[i]
            # Update gt class to cluster mapping
            if gt_class not in gt_to_cluster_intersection:
                gt_to_cluster_intersection[gt_class] = {predicted_cluster: 0}
            if predicted_cluster not in gt_to_cluster_intersection[gt_class]:
                gt_to_cluster_intersection[gt_class][predicted_cluster] = 0
            gt_to_cluster_intersection[gt_class][predicted_cluster] += 1
            # Update cluster to gt class mapping
            if predicted_cluster not in cluster_to_gt_intersection:
                cluster_to_gt_intersection[predicted_cluster] = {gt_class: 0}
            if gt_class not in cluster_to_gt_intersection[predicted_cluster]:
                cluster_to_gt_intersection[predicted_cluster][gt_class] = 0
            cluster_to_gt_intersection[predicted_cluster][gt_class] += 1

        purity = (1 / N) * sum([max(x.values()) for x in cluster_to_gt_intersection.values()])
        collocation = (1 / N) * sum([max(x.values()) for x in gt_to_cluster_intersection.values()])
        if purity + collocation == 0:
            f1 = 0
        else:
            f1 = 2 * (purity * collocation) / (purity + collocation)

        return purity, collocation, f1

    @staticmethod
    def calc_fscore(gt_labels, predicted_labels):
        N = len(gt_labels)
        gt_to_cluster_intersection = {}
        gt_class_count = {}
        cluster_count = {}
        for i in range(N):
            gt_class = gt_labels[i]
            predicted_cluster = predicted_labels[i]

            # Update counts
            if gt_class not in gt_class_count:
                gt_class_count[gt_class] = 0
            gt_class_count[gt_class] += 1
            if predicted_cluster not in cluster_count:
                cluster_count[predicted_cluster] = 0
            cluster_count[predicted_cluster] += 1

            # Update gt class to cluster mapping
            if gt_class not in gt_to_cluster_intersection:
                gt_to_cluster_intersection[gt_class] = {predicted_cluster: 0}
            if predicted_cluster not in gt_to_cluster_intersection[gt_class]:
                gt_to_cluster_intersection[gt_class][predicted_cluster] = 0
            gt_to_cluster_intersection[gt_class][predicted_cluster] += 1

        Fscore = 0
        for gt_class, cluster_map in gt_to_cluster_intersection.items():
            gt_class_size = gt_class_count[gt_class]
            cur_class_F = 0
            for cluster, intersection_size in cluster_map.items():
                cluster_size = cluster_count[cluster]
                precision = intersection_size / gt_class_size
                recall = intersection_size / cluster_size
                if precision + recall == 0:
                    f = 0
                else:
                    f = 2 * (precision * recall) / (precision + recall)
                if f > cur_class_F:
                    cur_class_F = f

            Fscore += (gt_class_size / N) * cur_class_F

        return Fscore

    def report(self):
        if self.results is None:
            self.calc_results()

        res = self.name_prefix_str + ': '
        res += 'v measure score: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_v_measure_score'] + ', '
        res += 'purity: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_purity'] + ', '
        res += 'collocation: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_collocation'] + ', '
        res += 'purity-collocation F1: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_pu_co_f1'] + ', '
        res += 'FScore: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_FScore'] + ', '
        res += 'fuzzy homogeneity: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_fuzzy_homogeneity'] + ', '
        res += 'fuzzy completeness: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_fuzzy_completeness'] + ', '
        res += 'fuzzy v measure score: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_fuzzy_v_measure_score']

        return res

    def calc_results(self):
        self.results = {}

        self.calculate_scatter_metric()
        self.calculate_fuzzy_scatter_metric()

        # Add unknown words prefix to all metric names
        keys_to_remove = []

        item_list = list(self.results.items())
        for key, val in item_list:
            keys_to_remove.append(key)
            self.results[self.name_prefix_str + '_' + key] = val

        for key in keys_to_remove:
            del self.results[key]

    def uses_external_dataset(self):
        return True


class ClusterCounterMetric(Metric):
    """ This metric counts how many active clusters we have.
        An active cluster is a cluster with at least one word that crosses the threshold. """

    def __init__(self, text_model, token_count):
        super(ClusterCounterMetric, self).__init__(None, text_model)

        self.token_count = token_count

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    def report(self):
        if self.results is None:
            self.calc_results()

        res = 'Used cluster number: ' + str(self.results['used_cluster_num'])
        return res

    def calc_results(self):
        self.results = {}
        used_cluster_indicators = [False] * self.text_model.config.cluster_num
        text_threshold = self.text_model.config.text_threshold

        for token in self.token_count.keys():
            prediction_res = self.text_model.underlying_model.predict_cluster(token)
            if prediction_res is not None:
                predicted_cluster, prob = prediction_res
                if prob >= text_threshold:
                    used_cluster_indicators[predicted_cluster] = True

        used_cluster_num = len([x for x in used_cluster_indicators if x is True])
        self.results['used_cluster_num'] = used_cluster_num

    def uses_external_dataset(self):
        return True

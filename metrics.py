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

    def document(self, orig_image_sizes, predicted_bboxes, gt_bboxes):
        iou_threshold = self.visual_model.config.object_threshold

        batch_size = len(gt_bboxes)
        for sample_ind in range(batch_size):
            gt_bbox_num = len(gt_bboxes[sample_ind])
            sample_gt_bboxes = gt_bboxes[sample_ind]
            sample_gt_bboxes = get_resized_gt_bboxes(sample_gt_bboxes, orig_image_sizes[sample_ind])
            sample_gt_bboxes = torch.stack([torch.tensor(x) for x in sample_gt_bboxes])
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

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_bboxes = self.visual_model.predict_bboxes(visual_inputs)
        gt_bboxes = visual_metadata['gt_bboxes']
        orig_image_sizes = visual_metadata['orig_image_size']
        self.document(orig_image_sizes, predicted_bboxes, gt_bboxes)

    def document_with_loaded_results(self, orig_image_sizes, activation_maps, gt_bboxes):
        predicted_bboxes = self.visual_model.predict_bboxes_with_activation_maps(activation_maps)
        self.document(orig_image_sizes, predicted_bboxes, gt_bboxes)

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

    def report(self):
        return self.report_with_name('Noun prediction results')


class ConcretenessPredictionMetric(Metric):
    """ This metric uses a concreteness dataset: a human-annotated dataset
    that gives every word a number between 1 and 5, representing how
    concrete is this word. We compare it to our prediction: a word that
    instantiates a concept is concrete (5) and is otherwise in-concrete (1).
    """

    def __init__(self, text_model, concreteness_dataset, token_count):
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
            concreteness_estimation = self.text_model.estimate_concept_instantiation_per_word([[token]])[0][0]
            concreteness_prediction = concreteness_estimation >= self.text_model.config.noun_threshold
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

    def report(self):
        self.traverse_vocab()

        estimation_mae = self.estimation_absolute_error_sum / self.tested_words_count
        estimation_mae_for_conc = self.estimation_absolute_error_sum_for_conc_words / self.conc_words_count
        estimation_mae_for_non_conc = self.estimation_absolute_error_sum_for_non_conc_words / self.non_conc_words_count

        prediction_mae = self.prediction_absolute_error_sum / self.tested_words_count
        prediction_mae_for_conc = self.prediction_absolute_error_sum_for_conc_words / self.conc_words_count
        prediction_mae_for_non_conc = self.prediction_absolute_error_sum_for_non_conc_words / self.non_conc_words_count

        min_count_to_correlation = {}
        for i in range(len(self.min_count_vals)):
            gt_and_estimations = np.array([self.min_count_gt_lists[i], self.min_count_est_lists[i]])
            gt_and_predictions = np.array([self.min_count_gt_lists[i], self.min_count_pred_lists[i]])
            est_pearson_corr = np.corrcoef(gt_and_estimations)[0, 1]
            pred_pearson_corr = np.corrcoef(gt_and_predictions)[0, 1]
            min_count_to_correlation[self.min_count_vals[i]] = \
                (est_pearson_corr, pred_pearson_corr, len(self.min_count_gt_lists[i]))

        return 'Concreteness estimation mean absolute error: ' + str(estimation_mae) + \
               ' overall, ' + str(estimation_mae_for_conc) + ' for concrete-predicted words, ' + \
               str(estimation_mae_for_non_conc) + ' for non-concrete-predicted words, ' + \
               'concreteness prediction mean absolute error: ' + str(prediction_mae) + \
               ' overall, ' + str(prediction_mae_for_conc) + ' for concrete-predicted words, ' + \
               str(prediction_mae_for_non_conc) + ' for non-concrete-predicted words, ' + \
               'pearson correlation by token count: ' + str(min_count_to_correlation)


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
        concepts_by_image = self.visual_model.predict_concept_indicators()
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


class VisualKnownClassesClassificationMetric(VisualClassificationMetric):
    """ This metric is only for models trained with labels of classes.
    We train a visual model to predict the classes on an image, and evaluate its predictions, given the ground
    truth classes. """

    def __init__(self, visual_model, class_num):
        super(VisualKnownClassesClassificationMetric, self).__init__(visual_model)
        self.class_num = class_num

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_classes = self.visual_model.predict_concept_lists()

        gt_classes_str = visual_metadata['gt_classes']
        gt_classes = [int(x) for x in gt_classes_str.split(',')]
        self.document(predicted_classes, gt_classes)

    def document(self, predicted_classes, gt_classes):
        self.evaluate_classification(predicted_classes, gt_classes)

    def report(self):
        return self.report_with_name('Visual classification results')


class VisualUnknownClassesClassificationMetric(VisualClassificationMetric):
    """ This metric is only for models that have labels of classes in their evaluation set, but were not trained
    with these labels.
    We train a visual model to cluster images, and evaluate its clustering by mapping the clusters to the
    labelled classes in a many-to-one manner, by maximizing the F1 score. """

    def __init__(self, visual_model, mapping_mode):
        super(VisualUnknownClassesClassificationMetric, self).__init__(visual_model)

        # Maintain a list of pairs of (predicted clusters, gt classes) for future calculations
        self.predicted_clusters_gt_classes = []
        ''' Mapping mode: Two possible heuristics for choosing concept to class mapping:
        First, for each concept choose the class with which it co-occurred the most.
        Second, for each concept choose the class with which it has the largest IoU. '''
        self.mapping_mode = mapping_mode

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_classes = self.visual_model.predict_concept_lists()
        self.document(predicted_classes, visual_metadata['gt_classes'])

    def document(self, predicted_classes, gt_classes):
        batch_size = len(gt_classes)
        self.predicted_clusters_gt_classes += \
            [(predicted_classes[i], gt_classes[i]) for i in range(batch_size)]

    def evaluate(self):
        # Get a unique list of concepts
        concepts_with_repetition = [x[0] for x in self.predicted_clusters_gt_classes]
        concept_list = list(set([inner for outer in concepts_with_repetition for inner in outer]))
        self.class_num = len(concept_list)

        # First, document for each concept how many times each class co-occurred with it
        concept_class_co_occur = {x: {} for x in concept_list}

        predicted_concept_count = {x: 0 for x in concept_list}
        gt_class_count = {}
        for predicted_concepts, gt_classes in self.predicted_clusters_gt_classes:
            for predicted_concept in predicted_concepts:
                # Increment count
                predicted_concept_count[predicted_concept] += 1
                # Go over gt classes
                for gt_class in gt_classes:
                    # Increment count
                    if gt_class not in gt_class_count:
                        gt_class_count[gt_class] = 0
                    gt_class_count[gt_class] += 1
                    # Document co-occurrence
                    if gt_class not in concept_class_co_occur[predicted_concept]:
                        concept_class_co_occur[predicted_concept][gt_class] = 0
                    concept_class_co_occur[predicted_concept][gt_class] += 1

        ''' Two possible heuristics for choosing concept to class mapping:
        First, for each concept choose the class with which it co-occurred the most.
        Second, for each concept choose the class with which it has the largest IoU. '''
        # First option: for each concept, choose the class with which it co-occurred the most
        if self.mapping_mode == 'co_occur':
            concept_to_class = {
                x: max(concept_class_co_occur[x], key=concept_class_co_occur[x].get)
                if len(concept_class_co_occur[x]) > 0 else None
                for x in concept_list
            }

            # Finally, go over the results again and use the mapping to evaluate
            for predicted_concepts, gt_classes in self.predicted_clusters_gt_classes:
                predicted_classes = [concept_to_class[x] for x in predicted_concepts]
                self.evaluate_classification([predicted_classes], [gt_classes])

            # Apart from the classification results, we want to measure the intersection of our classes and the gt classes
            intersections = {x: concept_class_co_occur[x][concept_to_class[x]] for x in concept_list}
            unions = {x: predicted_concept_count[x] +  # Concept count
                      gt_class_count[concept_to_class[x]] -  # Class count
                      intersections[x]  # Intersection count
                      for x in concept_list}
            self.ious = {x: intersections[x] / unions[x] if unions[x] > 0 else 0
                         for x in concept_list}

            concept_to_class = {
                x: max(concept_class_co_occur[x], key=concept_class_co_occur[x].get)
                if len(concept_class_co_occur[x]) > 0 else None
                for x in concept_list
            }
        # Second option: for each concept choose the class with which it has the largest IoU
        elif self.mapping_mode == 'iou':
            intersections = concept_class_co_occur
            unions = {
                concept_ind: {
                    class_ind:
                        predicted_concept_count[concept_ind] +  # Concept count
                        gt_class_count[class_ind] -  # Class count
                        intersections[concept_ind][class_ind]  # Intersection count
                    if class_ind in intersections[concept_ind] else 0
                    for class_ind in gt_class_count.keys()
                }
                for concept_ind in concept_list
            }
            ious = {
                concept_ind: {
                    class_ind:
                        intersections[concept_ind][class_ind] / unions[concept_ind][class_ind]
                        if unions[concept_ind][class_ind] > 0 else 0
                    for class_ind in gt_class_count.keys()
                }
                for concept_ind in concept_list
            }

            # Now, for each concept, choose the class with which it co-occurred the most
            concept_to_class = {
                x: max(ious[x], key=ious[x].get)
                if len(ious[x]) > 0 else None
                for x in concept_list
            }

            self.ious = {concept_ind: ious[concept_ind][concept_to_class[concept_ind]] for concept_ind in concept_list}

        # Finally, go over the results again and use the mapping to evaluate
        for predicted_concepts, gt_classes in self.predicted_clusters_gt_classes:
            predicted_classes = [concept_to_class[x] for x in predicted_concepts]
            self.evaluate_classification([predicted_classes], [gt_classes])

        self.concept_to_class = concept_to_class

    def report(self):
        """In this metric we have post analysis, we'll do it in the report function as this function is
        executed after all calculations are done."""
        self.evaluate()

        res = self.report_with_name('Visual classification results ' + str(self.mapping_mode) + ' mapping')
        res += ', iou max ' + str(max(self.ious.values())) + ' min ' + str(min(self.ious.values())) + \
               ' mean ' + str(statistics.mean(self.ious.values())) + ' median ' + \
               str(statistics.median(self.ious.values())) + '\n'
        res += 'Concept class pairs: '
        for concept_ind, iou in self.ious.items():
            class_ind = self.concept_to_class[concept_ind]
            res += '(' + str(concept_ind) + ',' + str(class_ind) + ',' + "{:.2f}".format(iou) + ') '

        return res


class VisualPromptClassificationMetric(VisualClassificationMetric):
    """ This metric is for models mapping both images and text to the same space.
    We prompt the model by giving it the name of the ground-truth classes, and this enables us to map the model's
    clusters to gt classes. """

    def __init__(self, visual_model, text_model, class_mapping):
        super(VisualPromptClassificationMetric, self).__init__(visual_model)
        self.text_model = text_model
        self.concept_to_gt_class = self.text_model.create_concept_to_gt_class_mapping(class_mapping)
        self.class_num = len(class_mapping)

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        batch_size = len(text_inputs)
        predicted_concepts = self.visual_model.predict_concept_lists()
        for sample_ind in range(batch_size):
            predicted_class_lists = [self.concept_to_gt_class[x] for x in predicted_concepts[sample_ind]]
            predicted_classes = [inner for outer in predicted_class_lists for inner in outer]

            gt_classes = visual_metadata['gt_classes'][sample_ind]
            self.document([predicted_classes], [gt_classes])

    def document(self, predicted_classes, gt_classes):
        self.evaluate_classification(predicted_classes, gt_classes)

    def report(self):
        return self.report_with_name('Prompt classification results')


class CategorizationMetric(Metric):
    """ This metric estimate whether the textual model categorizes words according to some baseline (the category
    dataset).
    We use the V-measure-score for evaluation. """

    def __init__(self, text_model, category_dataset):
        super(CategorizationMetric, self).__init__(None, text_model)
        self.category_dataset = category_dataset
        self.category_to_scatter_metric = {}

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        # This metric is not related to the test set, all the measurements are done later
        return

    def calculate_scatter_metric(self):
        gt_labels = []
        predicted_labels = []
        category_index = 0
        for category, word_list in self.category_dataset.items():
            for word in word_list:
                prediction = self.text_model.model.predict_concept(word)
                if prediction is not None:
                    gt_labels.append(category_index)
                    predicted_labels.append(prediction[0])
            category_index += 1
        self.v_measure_score = v_measure_score(gt_labels, predicted_labels)

    def report(self):
        self.calculate_scatter_metric()
        res = 'v measure score: ' + str(self.v_measure_score)

        return res

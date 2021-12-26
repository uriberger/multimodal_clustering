###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from metrics.sensitivity_specificity_metrics.sensitivity_specificity_metric import SensitivitySpecificityMetric


class VisualClassificationMetric(SensitivitySpecificityMetric):
    """ Base class for image multi-label classification metrics. """

    def __init__(self, visual_model, class_num):
        super(VisualClassificationMetric, self).__init__(visual_model, None)
        self.class_num = class_num

    def evaluate_classification(self, predicted_classes, gt_classes):
        batch_size = len(predicted_classes)
        for sample_ind in range(batch_size):
            sample_predicted = predicted_classes[sample_ind]
            sample_gt = gt_classes[sample_ind]
            cur_tp, cur_fp, cur_fn, cur_tn = self.calculate_sens_spec_metrics(sample_predicted, sample_gt)

            self.tp += cur_tp
            self.fp += cur_fp
            self.fn += cur_fn
            self.tn += cur_tn

    """ Calculate the number of true positive, false positive, false negatives, true negatives given a list of predicted
        classes and the list of ground-truth classes.
        For each class, we count it's occurrences in both lists, and use it to find these metrics.
    """

    def calculate_sens_spec_metrics(self, predicted_classes, gt_classes):
        predicted_class_to_count = self.count_class_instances(predicted_classes)
        gt_class_to_count = self.count_class_instances(gt_classes)

        all_classes = list(set(predicted_classes).union(gt_classes))
        false_negative_classes = []

        tp_count = 0
        fp_count = 0
        fn_count = 0
        for class_ind in all_classes:
            if class_ind in predicted_class_to_count:
                predicted_class_count = predicted_class_to_count[class_ind]
            else:
                predicted_class_count = 0

            if class_ind in gt_class_to_count:
                gt_class_count = gt_class_to_count[class_ind]
            else:
                gt_class_count = 0

            # The number of true positives is the minimum between number of predicted instances and number of gt
            # instances
            intersection_count = min(predicted_class_count, gt_class_count)
            tp_count += intersection_count

            if predicted_class_count > gt_class_count:
                # All predicted occurrences that are not in the ground-truth are false positive
                fp_count += predicted_class_count - gt_class_count
            if gt_class_count > predicted_class_count:
                # All ground-truth occurrences that were not predicted are false negative
                fn_count += gt_class_count - predicted_class_count
                false_negative_classes.append(class_ind)

        predicted_classes = list(set(predicted_classes))
        predicted_num = len(predicted_classes)
        non_predicted_num = self.class_num - predicted_num
        tn_count = non_predicted_num - len(false_negative_classes)

        return tp_count, fp_count, fn_count, tn_count

    @staticmethod
    def count_class_instances(inst_list):
        class_to_count = {}
        for class_inst in inst_list:
            if class_inst not in class_to_count:
                class_to_count[class_inst] = 0
            class_to_count[class_inst] += 1
        return class_to_count

    @abc.abstractmethod
    def document(self, predicted_classes, gt_classes):
        return

    @staticmethod
    def is_image_only():
        return True

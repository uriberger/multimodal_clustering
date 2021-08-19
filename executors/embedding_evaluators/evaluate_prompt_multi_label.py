from executors.embedding_evaluators.evaluate_prompt import PromptEvaluator
from torch.utils import data
from utils.general_utils import for_loop_with_reports
import torch


class PromptMultiLabelEvaluator(PromptEvaluator):
    """ Evaluate prompt for multi-label datasets. """

    def __init__(self, test_set, class_mapping, gt_classes_file, model_type, model_str, indent):
        super(PromptMultiLabelEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)

        self.gt_classes_data = torch.load(gt_classes_file)

    def metric_pre_calculations(self):
        """ In multi-label prompt evaluation, there's a heavy pre-calculation stage.
        We need to go over all the samples, and for each sample estimate its similarity with each class.
        Only later on we will choose the threshold of similarity to be the threshold that maximizes the final F1 score."""
        PromptEvaluator.metric_pre_calculations(self)

        ''' To choose the best threshold, we will run the following algorithm:
        First, for each sample, we will collect the similarity of the image embedding with each text class's embedding,
        and in addition whether this class is ground-truth or not (for this sample). Next, we will sort this list
        according to the similarity. Then we can say that for threshold x, all similarities smaller than x will be
        considered negative, so gt instances will be counted as false negative, and non-gt instanced will be counted
        as true negative (and equivalently for similarities larger than x with true positive and false positive).
        This will enable us to choose the best threshold for the F1 score. '''
        self.similarity_gt_list = []
        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=False)
        checkpoint_len = 10000
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.collect_similarity_and_gt, self.progress_report)
        self.decrement_indent()

        self.similarity_gt_list.sort(key=lambda x: x[0])

        self.choose_similarity_threshold()

    def collect_similarity_and_gt(self, sample_ind, sampled_batch, print_info):
        # Calculate similarity with each text class
        sample_embedding = self.embedding_mat[sample_ind, :]
        similarity_with_classes = {
            x: self.im_txt_similarity_func(sample_embedding, self.class_ind_to_embedding[x])
            for x in self.class_mapping.keys()
        }

        ''' Ground-truth classes and bboxes are not part of the dataset, because these are lists of varying
            length and thus cannot be batched using pytorch's data loader. So we need to extract these from an
            external file. '''
        image_id = sampled_batch['image_id'][0].item()
        gt_classes = self.gt_classes_data[image_id]

        for class_ind, similarity in similarity_with_classes.items():
            self.similarity_gt_list.append((similarity.item(), class_ind in gt_classes, sample_ind, class_ind))

    def choose_similarity_threshold(self):
        """ The similarity gt list is a sorted list of pairs of similarity of the embedding of the image in a specific
        sample to one of the text classes embedding, and an indicator whether this class is ground-truth for this
        sample.
        To choose the best similarity threshold, we need to know how many gt and non-gt there are before and after each
        element in the list.
        So, we go over the entire list twice: once to collect how many gt and non-gt there are before each element, and
        once to collect how many there are after each element.
        One thing we need to remember is that there might be multiple similarities with the same value in the list. So
        we need to update the count only after the last one with the same value. """
        similarities_num = len(self.similarity_gt_list)

        # First traverse
        gt_non_gt_count_before_element = self.collect_gt_non_gt_relative_to_element(False)

        # Second traverse
        gt_non_gt_count_after_element = self.collect_gt_non_gt_relative_to_element(True)

        # F1 calculation for each threshold
        best_F1 = -1
        for i in range(similarities_num):
            ''' In case we choose similarity number i to be the threshold, all the gt before it will be false negative,
            all non-gt before it will be true negative, all gt after it will be true positive, and all non-gt after it
            will be false positive. '''
            tp = gt_non_gt_count_after_element[i][0]
            fp = gt_non_gt_count_after_element[i][1]
            fn = gt_non_gt_count_before_element[i][0]
            f1 = tp / (tp + 0.5 * (fp + fn))  # This is the definition of F1
            if f1 > best_F1:
                best_threshold = self.similarity_gt_list[i][0]
                best_F1 = f1

        self.sample_ind_to_predicted_classes = {}
        for similarity, _, sample_ind, class_ind in self.similarity_gt_list:
            if sample_ind not in self.sample_ind_to_predicted_classes:
                self.sample_ind_to_predicted_classes[sample_ind] = []
            if similarity >= best_threshold:
                self.sample_ind_to_predicted_classes[sample_ind].append(class_ind)

    def collect_gt_non_gt_relative_to_element(self, reverse):
        similarities_num = len(self.similarity_gt_list)
        if reverse:
            ind_start = similarities_num - 1
            ind_end = -1
            step = -1
        else:
            ind_start = 0
            ind_end = similarities_num
            step = 1

        gt_non_gt_count_relative_to_element = []
        gt_count_so_far = 0
        non_gt_count_so_far = 0
        gt_count_from_last_different_similarity = 0
        non_gt_count_from_last_different_similarity = 0
        cur_similarity_count = 0
        for i in range(ind_start, ind_end, step):
            similarity, is_gt, _, _ = self.similarity_gt_list[i]
            if i == ind_start:
                prev_similarity = similarity

            if similarity != prev_similarity:
                if not reverse:
                    ''' In case we're going in normal direction, we don't want to include the gt and non-gt count of the
                     current similarity in the list. Also, we'll add the new count at the end of the list. '''
                    gt_non_gt_count_relative_to_element = \
                        gt_non_gt_count_relative_to_element + \
                        [(gt_count_so_far, non_gt_count_so_far)] * cur_similarity_count
                gt_count_so_far += gt_count_from_last_different_similarity
                non_gt_count_so_far += non_gt_count_from_last_different_similarity
                if reverse:
                    ''' In case we're going in reverse order, we want to include the gt and non-gt count of the current
                    similarity in the list. Also, we'll add the new count at the beginning of the list. '''
                    gt_non_gt_count_relative_to_element = \
                        [(gt_count_so_far, non_gt_count_so_far)] * cur_similarity_count + \
                        gt_non_gt_count_relative_to_element
                gt_count_from_last_different_similarity = 0
                non_gt_count_from_last_different_similarity = 0
                prev_similarity = similarity
                cur_similarity_count = 0

            if is_gt:
                gt_count_from_last_different_similarity += 1
            else:
                non_gt_count_from_last_different_similarity += 1

            cur_similarity_count += 1

        # In the end, we'll have the last batch of equal similarities, need to update for those as well
        if not reverse:
            gt_non_gt_count_relative_to_element = \
                gt_non_gt_count_relative_to_element + \
                [(gt_count_so_far, non_gt_count_so_far)] * cur_similarity_count
        else:
            gt_count_so_far += gt_count_from_last_different_similarity
            non_gt_count_so_far += non_gt_count_from_last_different_similarity
            gt_non_gt_count_relative_to_element = \
                [(gt_count_so_far, non_gt_count_so_far)] * cur_similarity_count + \
                gt_non_gt_count_relative_to_element

        return gt_non_gt_count_relative_to_element

    def predict_classes(self, sample_ind):
        return self.sample_ind_to_predicted_classes[sample_ind]

    def get_labels_from_batch(self, batch):
        image_id = batch['image_id'][0].item()
        gt_classes = self.gt_classes_data[image_id]
        return gt_classes

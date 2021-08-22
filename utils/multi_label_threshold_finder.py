""" The purpose of the functions in this file is to find, given a multi-label classification problem, and a model that
predicts the probability of each class for each sample, what is the best threshold, where classes with probability
lower than the threshold will be considered as negative, and classes with probability that exceeds the threshold will
be considered as positive.
The threshold is chosen to maximize the F1 score.
The input to the main function is a list of tuples: (probability, ground truth, sample index, class index).
For example, if sample no. 5 instantiates the "dog" class which is class no. 3, and we predicted that this class is
instantiated in this sample with probability 0.7, we'll get the tuple (0.7, True, 5, 3).
The output of the main function is a mapping from sample index to a list of class indices that are instantiated in this
sample, according to the best threshold we calculated internally. """


def generate_sample_to_predicted_classes_mapping(prob_gt_list):
    prob_gt_list.sort(key=lambda x: x[0])

    prob_threshold = choose_probability_threshold(prob_gt_list)

    sample_to_predicted_classes = {}
    for prob, _, sample_ind, class_ind in prob_gt_list:
        if sample_ind not in sample_to_predicted_classes:
            sample_to_predicted_classes[sample_ind] = []
        if prob >= prob_threshold:
            sample_to_predicted_classes[sample_ind].append(class_ind)

    return sample_to_predicted_classes


def choose_probability_threshold(prob_gt_list):
    """ To choose the best probability threshold, we need to know how many gt and non-gt there are before and after each
    element in the list.
    So, we go over the entire list twice: once to collect how many gt and non-gt there are before each element, and
    once to collect how many there are after each element.
    One thing we need to remember is that there might be multiple probabilities with the same value in the list. So
    we need to update the count only after the last one with the same value. """
    prob_num = len(prob_gt_list)

    # First traverse
    gt_non_gt_count_before_element = collect_gt_non_gt_relative_to_element(prob_gt_list, False)

    # Second traverse
    gt_non_gt_count_after_element = collect_gt_non_gt_relative_to_element(prob_gt_list, True)

    # F1 calculation for each threshold
    best_F1 = -1
    for i in range(prob_num):
        ''' In case we choose similarity number i to be the threshold, all the gt before it will be false negative,
        all non-gt before it will be true negative, all gt after it will be true positive, and all non-gt after it
        will be false positive. '''
        tp = gt_non_gt_count_after_element[i][0]
        fp = gt_non_gt_count_after_element[i][1]
        fn = gt_non_gt_count_before_element[i][0]
        f1 = tp / (tp + 0.5 * (fp + fn))  # This is the definition of F1
        if f1 > best_F1:
            best_threshold = prob_gt_list[i][0]
            best_F1 = f1

    return best_threshold


def collect_gt_non_gt_relative_to_element(prob_gt_list, reverse):
    prob_num = len(prob_gt_list)
    if reverse:
        ind_start = prob_num - 1
        ind_end = -1
        step = -1
    else:
        ind_start = 0
        ind_end = prob_num
        step = 1

    gt_non_gt_count_relative_to_element = []
    gt_count_so_far = 0
    non_gt_count_so_far = 0
    gt_count_from_last_different_prob = 0
    non_gt_count_from_last_different_prob = 0
    cur_prob_count = 0
    for i in range(ind_start, ind_end, step):
        prob, is_gt, _, _ = prob_gt_list[i]
        if i == ind_start:
            prev_prob = prob

        if prob != prev_prob:
            if not reverse:
                ''' In case we're going in normal direction, we don't want to include the gt and non-gt count of the
                 current probability in the list. Also, we'll add the new count at the end of the list. '''
                gt_non_gt_count_relative_to_element = \
                    gt_non_gt_count_relative_to_element + \
                    [(gt_count_so_far, non_gt_count_so_far)] * cur_prob_count
            gt_count_so_far += gt_count_from_last_different_prob
            non_gt_count_so_far += non_gt_count_from_last_different_prob
            if reverse:
                ''' In case we're going in reverse order, we want to include the gt and non-gt count of the current
                probability in the list. Also, we'll add the new count at the beginning of the list. '''
                gt_non_gt_count_relative_to_element = \
                    [(gt_count_so_far, non_gt_count_so_far)] * cur_prob_count + \
                    gt_non_gt_count_relative_to_element
            gt_count_from_last_different_prob = 0
            non_gt_count_from_last_different_prob = 0
            prev_prob = prob
            cur_prob_count = 0

        if is_gt:
            gt_count_from_last_different_prob += 1
        else:
            non_gt_count_from_last_different_prob += 1

        cur_prob_count += 1

    # In the end, we'll have the last batch of equal probabilities, need to update for those as well
    if not reverse:
        gt_non_gt_count_relative_to_element = \
            gt_non_gt_count_relative_to_element + \
            [(gt_count_so_far, non_gt_count_so_far)] * cur_prob_count
    else:
        gt_count_so_far += gt_count_from_last_different_prob
        non_gt_count_so_far += non_gt_count_from_last_different_prob
        gt_non_gt_count_relative_to_element = \
            [(gt_count_so_far, non_gt_count_so_far)] * cur_prob_count + \
            gt_non_gt_count_relative_to_element

    return gt_non_gt_count_relative_to_element

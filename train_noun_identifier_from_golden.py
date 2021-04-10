from flickr30 import generate_captions
from golden_semantic_class_dataset import generate_semantic_class_dataset, generate_noun_identification_dataset
from noun_identifier import NounIdentifier
import numpy as np
import time


def train_noun_identifier_from_golden():
    img_to_caption_list = generate_captions()
    caption_list = []
    for cur_list in img_to_caption_list.values():
        caption_list += cur_list

    caption_num = len(caption_list)
    cur_perm = np.random.permutation(caption_num)
    train_sample_num = int(0.9 * caption_num)
    eval_sample_num = caption_num - train_sample_num
    train_inds = cur_perm[:train_sample_num]
    eval_inds = cur_perm[train_sample_num:]
    train_caption_list = [caption_list[x] for x in train_inds]
    eval_caption_list = [caption_list[x] for x in eval_inds]

    training_set, semantic_class_num = generate_semantic_class_dataset(train_caption_list)

    noun_identifier = NounIdentifier(semantic_class_num)

    print('Training noun identifier from golden dataset...')
    checkpoint_len = 10000
    checkpoint_time = time.time()
    for i in range(train_sample_num):
        if i % checkpoint_len == 0:
            time_from_prev_checkpoint = time.time() - checkpoint_time
            checkpoint_time = time.time()
            print('\tStarting sample ' + str(i) + ' out of ' + str(train_sample_num) + ', time from prev checkpoint ' + str(
                time_from_prev_checkpoint))

        caption = training_set[i][0]
        semantic_class_inds = training_set[i][1]

        for token in caption:
            for semantic_class_ind in semantic_class_inds:
                noun_identifier.document_co_occurence(token, semantic_class_ind)
    print('Finished training noun identifier')

    eval_set, eval_token_num = generate_noun_identification_dataset(eval_caption_list)
    print('Evaluating noun identifier...')
    noun_threshold = 0.155
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0
    checkpoint_len = 1000
    checkpoint_time = time.time()
    for i in range(eval_sample_num):
        if i % checkpoint_len == 0:
            time_from_prev_checkpoint = time.time() - checkpoint_time
            checkpoint_time = time.time()
            print('\tStarting caption ' + str(i) + ' out of ' + str(
                eval_sample_num) + ', time from prev checkpoint ' + str(
                time_from_prev_checkpoint))

        caption = eval_set[i][0]
        noun_inds = eval_set[i][1]

        for j in range(len(caption)):
            token = caption[j]
            semantic_class_prediction = noun_identifier.predict_semantic_class(token)
            if semantic_class_prediction is None:
                # Never encountered this word before, consider this an incorrect prediction
                if noun_ground_truth:
                    fn_count += 1
                else:
                    fp_count += 1
            else:
                _, probability = semantic_class_prediction
                if probability >= noun_threshold:
                    noun_prediction = True
                else:
                    noun_prediction = False
                noun_ground_truth = (j in noun_inds)

                if noun_prediction and noun_ground_truth:
                    tp_count += 1
                elif noun_prediction and not noun_ground_truth:
                    fp_count += 1
                elif not noun_prediction and noun_ground_truth:
                    fn_count += 1
                elif not noun_prediction and not noun_ground_truth:
                    tn_count += 1
    print('Finished evaluating noun identifier')
    print('Eval sample num: ' + str(eval_token_num))
    print('True positive count: ' + str(tp_count))
    print('False positive count: ' + str(fp_count))
    print('True negative count: ' + str(tn_count))
    print('False negative count: ' + str(fn_count))

    accuracy = (tp_count + tn_count)/eval_token_num
    precision = tp_count/(tp_count + fp_count)
    recall = tp_count/(tp_count + fn_count)
    f1 = tp_count/(tp_count + 0.5*(fn_count + fp_count))

    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

import nltk
import torch
import time

noun_tags = [
    'NN',
    'NNP',
    'NNS',
    'NNPS',
    'PRP'
]


def is_noun(pos_tag):
    return pos_tag in noun_tags


def generate_semantic_class_dataset(caption_list):
    unique_nouns = {}
    noun_lists = []
    caption_num = len(caption_list)
    i = 0

    print('Generating noun list...')
    checkpoint_len = 100000
    checkpoint_time = time.time()
    for caption in caption_list:
        if i % checkpoint_len == 0:
            time_from_prev_checkpoint = time.time() - checkpoint_time
            checkpoint_time = time.time()
            print('\tStarting caption ' + str(i) + ' out of ' + str(caption_num) + ', time from prev checkpoint ' + str(
                time_from_prev_checkpoint))

        noun_lists.append([])
        pos_tag_result = nltk.pos_tag(caption)
        for token, tag in pos_tag_result:
            if is_noun(tag):
                unique_nouns[token] = True
                noun_lists[-1].append(token)

        i += 1
    print('Noun list generated')

    unique_noun_num = len(unique_nouns)
    print('Found ' + str(unique_noun_num) + ' nouns')

    ind_to_noun = list(unique_nouns.keys())
    noun_to_ind = {ind_to_noun[x]: x for x in range(unique_noun_num)}

    dataset = []
    print('Generating dataset...')
    checkpoint_len = 10000
    checkpoint_time = time.time()
    i = 0
    for noun_list in noun_lists:
        if i % checkpoint_len == 0:
            time_from_prev_checkpoint = time.time() - checkpoint_time
            checkpoint_time = time.time()
            print('\tStarting caption ' + str(i) + ' out of ' + str(caption_num) + ', time from prev checkpoint ' + str(
                time_from_prev_checkpoint))

        semantic_class_inds = [noun_to_ind[token] for token in noun_list]

        dataset.append((caption_list[i], semantic_class_inds))
        i += 1
    print('Dataset generated')

    return dataset, unique_noun_num


def generate_noun_identification_dataset(caption_list):
    dataset = []
    caption_num = len(caption_list)
    i = 0

    print('Generating dataset...')
    token_count = 0
    noun_count = 0
    checkpoint_len = 10000
    checkpoint_time = time.time()
    for caption in caption_list:
        if i % checkpoint_len == 0:
            time_from_prev_checkpoint = time.time() - checkpoint_time
            checkpoint_time = time.time()
            print('\tStarting caption ' + str(i) + ' out of ' + str(caption_num) + ', time from prev checkpoint ' + str(
                time_from_prev_checkpoint))

        pos_tag_result = nltk.pos_tag(caption)
        noun_inds = []
        j = 0
        for token, tag in pos_tag_result:
            if is_noun(tag):
                noun_inds.append(j)
                noun_count += 1
            j += 1
            token_count += 1

        dataset.append((caption, noun_inds))
        i += 1
    print('Dataset generated')
    print('Dataset contains ' + str(token_count) + ' tokens, ' + str(noun_count) + ' nouns which are '
          + str(noun_count/token_count) + ' percentage')

    return dataset, token_count

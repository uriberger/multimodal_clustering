from utils.general_utils import generate_dataset, for_loop_with_reports
import io
import os
import numpy as np

fast_text_dataset_filename = os.path.join('cached_dataset_files', 'fast_text')
fast_text_filename = 'wiki-news-300d-1M.vec'


def generate_fast_text(word_list):
    return generate_dataset(fast_text_dataset_filename,
                            generate_fast_text_internal,
                            fast_text_filename,
                            word_list)


def generate_fast_text_internal(dataset_filename, word_list):
    print('Generating fast text vectors dataset...')

    fin = io.open(dataset_filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    counter = 0
    for line in fin:
        if counter % 10000 == 0:
            print('Starting token ' + str(counter))
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word_list:
            data[word] = np.array([float(x) for x in tokens[1:]])
        counter += 1
    print('GOT HERE and found ' + str(len(data)) + ' words')
    return data

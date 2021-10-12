from utils.general_utils import generate_dataset
import os

cache_dirname = 'cached_dataset_files'
category_dataset_filename = os.path.join(cache_dirname, 'category_dataset')
extended_category_dataset_filename_prefix = os.path.join(cache_dirname, 'extended_category_dataset_')
category_filename = 'multilingual_concepts_categories.tsv'
extended_category_filename_prefix = 'category_dataset_'
extended_category_filename_suffix = '.txt'


def generate_category_dataset():
    return generate_dataset(category_dataset_filename,
                            generate_category_dataset_internal,
                            category_filename)


def generate_category_dataset_internal(dataset_filename):
    print('Generating category dataset...')

    fp = open(dataset_filename)
    cur_category = None
    new_category = True
    category_to_words = {}
    for line in fp:
        stripped_line = line.strip()
        if len(stripped_line) == 0:
            new_category = True
        else:
            english_word = stripped_line.split()[0]
            if new_category:
                cur_category = english_word
                category_to_words[cur_category] = []
                new_category = False
            else:
                category_to_words[cur_category].append(english_word)

    return category_to_words


def generate_extended_category_dataset(split):
    output_filename = extended_category_dataset_filename_prefix + split
    input_filename = extended_category_filename_prefix + split + extended_category_filename_suffix
    return generate_dataset(output_filename,
                            generate_extended_category_dataset_internal,
                            input_filename)


def generate_extended_category_dataset_internal(dataset_filename):
    print('Generating category dataset...')

    fp = open(dataset_filename)
    cur_category = None
    new_category = True
    category_to_words = {}
    for line in fp:
        stripped_line = line.strip()
        if len(stripped_line) == 0:
            new_category = True
        else:
            token = stripped_line
            if new_category:
                cur_category = token
                category_to_words[cur_category] = []
                new_category = False
            else:
                category_to_words[cur_category].append(token)

    return category_to_words

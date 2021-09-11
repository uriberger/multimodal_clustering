from utils.general_utils import generate_dataset
import os

category_dataset_filename = os.path.join('cached_dataset_files', 'category_dataset')
category_filename = 'multilingual_concepts_categories.tsv'


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

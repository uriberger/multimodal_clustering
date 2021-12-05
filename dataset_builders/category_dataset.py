from utils.general_utils import generate_dataset
from dataset_builders.dataset_builder import datasets_dir, cached_dataset_files_dir
import os
import yaml


category_dataset_filename = os.path.join(cached_dataset_files_dir, 'category_dataset')
extended_category_dataset_filename_prefix = os.path.join(cached_dataset_files_dir, 'extended_category_dataset_')
fountain_dataset_filename = os.path.join(cached_dataset_files_dir, 'fountain_dataset')
category_filename = 'multilingual_concepts_categories.tsv'
extended_category_filename_prefix = 'category_dataset_'
extended_category_filename_suffix = '.txt'
fountain_filename = os.path.join(datasets_dir, 'mcrae_typicality.yaml')


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


def generate_fountain_category_dataset():
    return generate_dataset(fountain_dataset_filename, generate_fountain_category_dataset_internal, fountain_filename)


def generate_fountain_category_dataset_internal(dataset_filename):
    print('Generating category dataset...')

    with open(dataset_filename) as f:
        base_category_to_word = yaml.load(f, Loader=yaml.FullLoader)
        word_lists = [list(x.keys()) for x in base_category_to_word.values()]
        all_words = list(set([word for outer in word_lists for word in outer]))
        word_to_categories = {word: {
            x[0]: x[1][word] for x in base_category_to_word.items() if word in x[1]
        } for word in all_words}
        word_to_category = {x[0]: max(x[1], key=x[1].get) for x in word_to_categories.items()}
        all_categories = list(word_to_category.values())
        category_to_word_list = {categ: [x[0] for x in word_to_category.items() if x[1] == categ]
                                 for categ in all_categories}

        return category_to_word_list

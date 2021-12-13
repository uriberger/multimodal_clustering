###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from utils.general_utils import generate_dataset
from dataset_builders.dataset_builder import datasets_dir, cached_dataset_files_dir
import os
import yaml

""" The methods in this builds the category dataset.
    The category dataset maps categories (e.g., 'fruit') to a list of words in these categories (e.g., 'banana',
    'apple').
"""

output_filename = os.path.join(cached_dataset_files_dir, 'fountain_dataset')
input_filename = os.path.join(datasets_dir, 'mcrae_typicality.yaml')


def generate_fountain_category_dataset():
    return generate_dataset(output_filename, generate_fountain_category_dataset_internal, input_filename)


def generate_fountain_category_dataset_internal(dataset_input_filename):
    print('Generating category dataset...')

    with open(dataset_input_filename) as f:
        ''' The input file maps categories to a dictionary of word: typicality rating, for example:
            {
                'reptile': {'iguana': 5.9, 'tortoise': 5.7, ...},
                'device': {'key': 4.6, 'radio': 5.0, ...}
            }
            A word may appear in multiple categories. For each word, we need to map it to the category in which it is
            most typical. 
        '''

        # First, create a dictionary of category: typicality rating for each word
        base_category_to_word = yaml.load(f, Loader=yaml.FullLoader)
        word_lists = [list(x.keys()) for x in base_category_to_word.values()]
        all_words = list(set([word for outer in word_lists for word in outer]))
        word_to_categories = {word: {
            x[0]: x[1][word] for x in base_category_to_word.items() if word in x[1]
        } for word in all_words}

        # Now, for each word, take the category with the highest typicality rating
        word_to_category = {x[0]: max(x[1], key=x[1].get) for x in word_to_categories.items()}

        # Finally, create the reversed mapping: categories to a list of words
        all_categories = list(word_to_category.values())
        category_to_word_list = {categ: [x[0] for x in word_to_category.items() if x[1] == categ]
                                 for categ in all_categories}

        return category_to_word_list

###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from utils.general_utils import generate_dataset
from dataset_builders.dataset_builder import DatasetBuilder
import os


class WordSimDatasetBuilder(DatasetBuilder):
    """ This class builds the word similarity dataset.
        The category dataset maps categories (e.g., 'fruit') to a list of words in these categories (e.g., 'banana',
        'apple').
        This file assumed the category dataset by Fountain and Lapata, presented in the paper
        "Meaning representation in natural language categorization".
    """

    def __init__(self, relatedness, indent):
        super(WordSimDatasetBuilder, self).__init__(indent)

        if relatedness:
            self.name = 'relatedness'
        else:
            self.name = 'similarity'

        self.output_file_path = os.path.join(self.cached_dataset_files_dir, self.name + '_dataset')

        input_dir_path = os.path.join(self.datasets_dir, 'wordsim353_sim_rel')

        self.input_file_path = os.path.join(input_dir_path, 'wordsim_' + self.name + '_goldstandard.txt')

    def build_dataset(self, config=None):
        return generate_dataset(self.output_file_path, self.generate_dataset_internal)

    def generate_dataset_internal(self):
        self.log_print('Generating ' + self.name + ' dataset...')

        dataset = []
        with open(self.input_file_path) as f:
            ''' The input file maps categories to a dictionary of word: typicality rating, for example:
                {
                    'reptile': {'iguana': 5.9, 'tortoise': 5.7, ...},
                    'device': {'key': 4.6, 'radio': 5.0, ...}
                }
                A word may appear in multiple categories. For each word, we need to map it to the category in which it is
                most typical. 
            '''

            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) != 3:
                    self.log_print('The following line doesn\'t have 3 tokens:')
                    self.log_print(line)
                    assert False
                dataset.append((line_parts[0], line_parts[1], float(line_parts[2])))

        return dataset

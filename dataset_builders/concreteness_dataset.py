from utils.general_utils import generate_dataset, for_loop_with_reports
from loggable_object import LoggableObject
import os

concreteness_dataset_filename = os.path.join('cached_dataset_files', 'concreteness_dataset')
concreteness_filename = 'Concreteness_ratings_Brysbaert_et_al_BRM.txt'


def generate_concreteness_dataset():
    return generate_dataset(concreteness_dataset_filename,
                            generate_concreteness_dataset_internal,
                            concreteness_filename)


class ConcretenessCollector(LoggableObject):

    def __init__(self, indent):
        super(ConcretenessCollector, self).__init__(indent)
        self.concreteness_dataset = {}

    def collect_line(self, index, line, print_info):
        if index == 0:
            return

        split_line = line.split()
        for token in split_line:
            if len(token) > 0 and not token[0].isalpha():
                break
        bigram_indicator = int(token)
        if bigram_indicator == 1:
            word = split_line[0] + ' ' + split_line[1]
            concreteness = float(split_line[3])
        else:
            word = split_line[0]
            concreteness = float(split_line[2])

        self.concreteness_dataset[word] = concreteness

    def file_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting line ' + str(index) + ', time from previous checkpoint ' + str(time_from_prev))


def generate_concreteness_dataset_internal(dataset_filename):
    print('Generating concreteness dataset...')

    collector = ConcretenessCollector(1)
    concreteness_fp = open(dataset_filename, 'r')
    checkpoint_len = 10000
    for_loop_with_reports(concreteness_fp, None, checkpoint_len,
                          collector.collect_line, collector.file_progress_report)

    return collector.concreteness_dataset

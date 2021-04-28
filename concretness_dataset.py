from aux_functions import generate_dataset
import time

concretness_dataset_filename = 'concretness_dataset'
concretness_filename = 'Concreteness_ratings_Brysbaert_et_al_BRM.txt'


def generate_concretness_dataset():
    return generate_dataset(concretness_dataset_filename, generate_concretness_dataset_internal,
                            concretness_filename)


def generate_concretness_dataset_internal(concretness_filename):
    print('Generating concretness dataset...')
    
    concretness_fp = open(concretness_filename, 'r')
    first = True
    count = 0
    concretness_dataset = {}
    checkpoint_len = 10000
    checkpoint_time = time.time()
    for line in concretness_fp:
        if first:
            first = False
            continue

        if count % checkpoint_len == 0:
            print('Starting line ' + str(count) + ', time from previous checkpoint ' + str(time.time() - checkpoint_time))
            checkpoint_time = time.time()
        
        split_line = line.split()
        word = split_line[0]
        concretness = float(split_line[2])
        concretness_dataset[word] = concretness

        count += 1

    return concretness_dataset

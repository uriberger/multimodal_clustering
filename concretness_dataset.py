from aux_functions import generate_dataset

concretness_dataset_filename = 'concretness_dataset'
concretness_filename = 'Concreteness_ratings_Brysbaert_et_al_BRM.txt'


def generate_concretness_dataset():
    return generate_dataset(concretness_dataset_filename, generate_concretness_dataset_internal,
                            concretness_filename)


def generate_concretness_dataset_internal(concretness_filename):
    concretness_fp = open(concretness_filename, 'r')
    for line in concretness_fp:
        print(line)
        assert False

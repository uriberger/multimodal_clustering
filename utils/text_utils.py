noun_tags = [
    'NN',
    'NNP',
    'NNS',
    'NNPS',
    'PRP'
]


def is_noun(pos_tag):
    return pos_tag in noun_tags


def preprocess_token(token):
    token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!"))
    token = token.lower()

    return token


def prepare_data(captions):
    token_lists = []
    for caption in captions:
        token_list = caption.split()
        token_list = [preprocess_token(token) for token in token_list]
        token_lists.append(token_list)
    return token_lists


def multiple_word_string(my_str):
    return len(my_str.split()) > 1

import torch.nn as nn
import spacy
nlp = spacy.load('en_core_web_sm')
tokenizer = nlp.tokenizer


noun_tags = [
    'NN',
    'NNP',
    'NNS',
    'NNPS',
    'PRP'
]

verb_tags = [
    'VBZ',
    'VB',
    'VBP',
    'VBG',
    'VBN',
    'VBD'
]

adjective_tags = [
    'JJ',
    'JJR',
    'JJS'
]


def is_noun(pos_tag):
    return pos_tag in noun_tags


def preprocess_token(token):
    token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!"))
    token = token.lower()

    return token


def prepare_data(captions):
    res = []
    for caption in captions:
        token_list = [str(x) for x in list(tokenizer(caption.lower()))]
        char_list = ''.join(token_list)
        res.append(char_list)
    return res


def multiple_word_string(my_str):
    return len(my_str.split()) > 1


def generate_text_model(model_str, output_size, word_embed_dim):
    # We don't know what the size of the vocabulary will be, so let's take some large value
    vocab_size = 50000
    num_layers = 2

    if model_str == 'lstm':
        model = nn.Sequential(
            nn.Embedding(vocab_size, word_embed_dim),
            nn.LSTM(word_embed_dim, output_size, num_layers, batch_first=True)
        )
    elif model_str == 'gru':
        model = nn.Sequential(
            nn.Embedding(vocab_size, word_embed_dim),
            nn.GRU(word_embed_dim, output_size, num_layers, batch_first=True)
        )

    return model

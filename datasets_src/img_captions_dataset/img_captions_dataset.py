import torch.utils.data as data
from utils.text_utils import prepare_data
import abc


class ImageCaptionDataset(data.Dataset):

    def __init__(self, config):
        super(ImageCaptionDataset, self).__init__()
        self.config = config

    def prepare_data(self, captions):
        return prepare_data(captions)

    @abc.abstractmethod
    def get_caption_data(self):
        return

    def get_token_count(self):
        token_count = {}
        caption_data = self.get_caption_data()
        i = 0
        for x in caption_data:
            token_list = self.prepare_data([x['caption']])[0]
            for token in token_list:
                if token not in token_count:
                    token_count[token] = 0
                token_count[token] += 1
            i += 1

        return token_count

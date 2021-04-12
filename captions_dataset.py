import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from coco import get_image_path
from PIL import Image
import numpy as np


class ImageCaptionDataset(data.Dataset):
    """A dataset with image samples and corresponding captions."""

    def __init__(self, wanted_image_size, captions_file_path, slice_str):
        self.image_id_caption_dataset = []
        self.caption_data = torch.load(captions_file_path)
        self.wanted_image_size = wanted_image_size
        self.slice_str = slice_str
        mean_tuple = (0.471, 0.4474, 0.4078)
        std_tuple = (0.2718, 0.2672, 0.2826)
        self.normalizer = transforms.Normalize(mean_tuple, std_tuple)

    def transform_to_torch_format(self, image_id):
        image_obj = Image.open(get_image_path(image_id, self.slice_str))
        # TODO: should I keep the next line? Is the resizing necessary?
        image_obj = image_obj.resize(self.wanted_image_size)
        image_tensor = torch.from_numpy(np.array(image_obj))/255
        image_tensor = image_tensor.permute(2, 0, 1)
        # image_tensor = self.normalizer(image_tensor)

        return image_tensor.float()

    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_caption_data = self.caption_data[idx]

        # Image
        image_id = item_caption_data['image_id']
        image_tensor = self.transform_to_torch_format(image_id)

        # Caption
        caption = item_caption_data['caption']

        sample = {'image': image_tensor, 'caption': caption}

        return sample

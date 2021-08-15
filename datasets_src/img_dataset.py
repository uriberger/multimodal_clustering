import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from utils.visual_utils import pil_image_trans
from PIL import Image


class ImageDataset(data.Dataset):
    """A dataset with only image samples.
    """

    def __init__(self,
                 caption_file,
                 get_image_path_func,
                 config):
        self.config = config

        mean_tuple = (0.471, 0.4474, 0.4078)
        std_tuple = (0.2718, 0.2672, 0.2826)
        self.normalizer = transforms.Normalize(mean_tuple, std_tuple)

        caption_data = torch.load(caption_file)
        self.image_ids = list(set([x['image_id'] for x in caption_data]))

        self.get_image_path_func = get_image_path_func

        self.transforms = transforms.Compose([
            # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
        ])

    def __len__(self):
        return len(self.image_ids)

    def slice_dataset(self, start_index, end_index):
        self.image_ids = self.image_ids[start_index:end_index]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.image_ids[idx]
        image_obj = Image.open(self.get_image_path_func, (image_id, self.config.slice_str))
        orig_image_size = image_obj.size
        image_tensor = pil_image_trans(image_obj)

        if self.config.use_transformations:
            image_tensor = self.transforms(image_tensor)

        sample = {
            'image_id': image_id,
            'image': image_tensor,
            'orig_image_size': orig_image_size,
            'index': idx
        }

        return sample

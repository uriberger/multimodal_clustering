import torch.utils.data as data
import torch
from coco import get_image_path
from PIL import Image
import numpy as np


class MultiLabelDataset(data.Dataset):
    """A dataset with image samples and class multi-label labelling."""

    def __init__(self, img_bboxes_dataset, class_num, wanted_image_size, slice_str):
        # Create id -> gt multi label dataset
        self.image_id_to_gt_classes_dataset = []
        for image_id, bbox_data in img_bboxes_dataset.items():
            gt_class_labels = [x[1] for x in bbox_data]
            unique_gt_class_labels = list(set(gt_class_labels))

            gt_label_vec = torch.zeros(class_num)
            gt_label_vec[torch.tensor(unique_gt_class_labels)] = 1.0

            self.image_id_to_gt_classes_dataset.append((image_id, gt_label_vec))

        self.wanted_image_size = wanted_image_size
        self.slice_str = slice_str

    def transform_to_torch_format(self, image_id):
        image_obj = Image.open(get_image_path(image_id, self.slice_str))
        # TODO: should I keep the next line? Is the resizing necessary?
        image_obj = image_obj.resize(self.wanted_image_size)
        image_tensor = torch.from_numpy(np.array(image_obj))

        output_image = image_tensor.permute(2, 0, 1)

        return output_image.float()

    def __len__(self):
        return len(self.image_id_to_gt_classes_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id, gt_label_vec = self.image_id_to_gt_classes_dataset[idx]
        image_tensor = self.transform_to_torch_format(image_id)
        sample = {'image': image_tensor, 'label': gt_label_vec}

        return sample

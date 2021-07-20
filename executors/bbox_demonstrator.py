from executors.demonstrator import Demonstrator
from models_src.visual_model_wrapper import VisualModelWrapper
import os
import torch
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
from models_src.model_config import wanted_image_size
import matplotlib.pyplot as plt
from utils.visual_utils import get_resized_gt_bboxes


class BboxDemonstrator(Demonstrator):

    def __init__(self, model_name, dataset, gt_classes_file_path, gt_bboxes_file_path, class_mapping,
                 num_of_items_to_demonstrate, indent):
        super(BboxDemonstrator, self).__init__(dataset, num_of_items_to_demonstrate, indent)

        visual_model_dir = os.path.join(self.models_dir, 'visual')
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, indent + 1, model_name)
        self.visual_model.eval()
        self.gt_classes_data = torch.load(gt_classes_file_path)
        self.gt_bboxes_data = torch.load(gt_bboxes_file_path)
        self.class_mapping = class_mapping

    def demonstrate_item(self, index, sampled_batch, print_info):
        image_tensor = sampled_batch['image']
        image_id = sampled_batch['image_id'].item()

        predicted_bboxes = self.visual_model.predict_bboxes(image_tensor)[0]
        gt_bboxes = get_resized_gt_bboxes(self.gt_bboxes_data[image_id], sampled_batch['orig_image_size'])
        gt_classes = self.gt_classes_data[image_id]

        image_obj = to_pil_image(image_tensor.view(3, wanted_image_size[0], wanted_image_size[1]))
        draw = ImageDraw.Draw(image_obj)

        # First draw ground-truth boxes
        for bbox_ind in range(len(gt_bboxes)):
            bbox = gt_bboxes[bbox_ind]
            gt_class = gt_classes[bbox_ind]
            draw.rectangle(bbox, outline=(255, 0, 0))
            text_loc = (bbox[0], bbox[1])
            draw.text(text_loc, self.class_mapping[gt_class], fill=(255, 0, 0))

        # Next, draw predicted boxes
        for bbox_ind in range(len(predicted_bboxes)):
            bbox = predicted_bboxes[bbox_ind]
            # gt_class = gt_classes[bbox_ind]
            draw.rectangle(bbox, outline=(0, 255, 0))
            # text_loc = (bbox[0], bbox[1])
            # draw.text(text_loc, self.class_mapping[gt_class], fill=(0, 255, 0))

        plt.imshow(image_obj)
        plt.show()

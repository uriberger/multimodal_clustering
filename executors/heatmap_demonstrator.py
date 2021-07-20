from executors.demonstrator import Demonstrator
from models_src.visual_model_wrapper import VisualModelWrapper
import os


class HeatmapDemonstrator(Demonstrator):

    def __init__(self, model_name, dataset, class_mapping,
                 num_of_items_to_demonstrate, indent):
        super(HeatmapDemonstrator, self).__init__(dataset, num_of_items_to_demonstrate, indent)

        visual_model_dir = os.path.join(self.models_dir, 'visual')
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, indent + 1, model_name)
        self.visual_model.eval()
        self.class_mapping = class_mapping

    def demonstrate_item(self, index, sampled_batch, print_info):
        image_tensor = sampled_batch['image']
        self.visual_model.plot_heatmap(image_tensor)

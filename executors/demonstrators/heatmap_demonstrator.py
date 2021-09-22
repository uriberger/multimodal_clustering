from executors.demonstrators.demonstrator import Demonstrator
from models_src.visual_model_wrapper import VisualModelWrapper
from models_src.textual_model_wrapper import generate_textual_model
import os
from utils.general_utils import visual_dir, text_dir


class HeatmapDemonstrator(Demonstrator):

    def __init__(self, model_name, dataset, class_mapping,
                 num_of_items_to_demonstrate, indent):
        super(HeatmapDemonstrator, self).__init__(dataset, num_of_items_to_demonstrate, indent)

        visual_model_dir = os.path.join(self.models_dir, visual_dir)
        textual_model_dir = os.path.join(self.models_dir, text_dir)
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_name, indent + 1)
        self.visual_model.eval()
        self.text_model = \
            generate_textual_model(self.device, 'counts_generative', textual_model_dir, model_name, indent + 1)

        gt_class_to_concept = {i: self.text_model.model.predict_concept(class_mapping[i])[0]
                               for i in range(len(class_mapping))
                               if ' ' not in class_mapping[i]}
        concept_to_gt_class_ind = {}
        for gt_class_ind, concept_ind in gt_class_to_concept.items():
            if concept_ind not in concept_to_gt_class_ind:
                concept_to_gt_class_ind[concept_ind] = []
            concept_to_gt_class_ind[concept_ind].append(gt_class_ind)

        concept_to_gt_class_str = {x: [class_mapping[i] for i in concept_to_gt_class_ind[x]]
                                   for x in concept_to_gt_class_ind.keys()}

        self.concept_to_gt_class_str = concept_to_gt_class_str

    def demonstrate_item(self, index, sampled_batch, print_info):
        image_tensor = sampled_batch['image']
        self.visual_model.plot_heatmap(image_tensor, self.concept_to_gt_class_str)

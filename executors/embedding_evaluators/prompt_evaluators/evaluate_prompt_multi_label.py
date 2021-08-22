from executors.embedding_evaluators.prompt_evaluators.evaluate_prompt import PromptEvaluator
from torch.utils import data
from utils.general_utils import for_loop_with_reports
from utils.multi_label_threshold_finder import generate_sample_to_predicted_classes_mapping
import torch


class PromptMultiLabelEvaluator(PromptEvaluator):
    """ Evaluate prompt for multi-label datasets. """

    def __init__(self, test_set, class_mapping, gt_classes_file, model_type, model_str, indent):
        super(PromptMultiLabelEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)

        self.gt_classes_data = torch.load(gt_classes_file)
        self.multi_label = True

    def metric_pre_calculations(self):
        """ In multi-label prompt evaluation, there's a heavy pre-calculation stage.
        We need to go over all the samples, and for each sample estimate its similarity with each class.
        Only later on we will choose the threshold of similarity to be the threshold that maximizes the final F1 score."""
        PromptEvaluator.metric_pre_calculations(self)

        ''' To choose the best threshold, we will run the following algorithm:
        First, for each sample, we will collect the similarity of the image embedding with each text class's embedding,
        and in addition whether this class is ground-truth or not (for this sample). Next, we will sort this list
        according to the similarity. Then we can say that for threshold x, all similarities smaller than x will be
        considered negative, so gt instances will be counted as false negative, and non-gt instanced will be counted
        as true negative (and equivalently for similarities larger than x with true positive and false positive).
        This will enable us to choose the best threshold for the F1 score. '''
        self.similarity_gt_list = []
        dataloader = data.DataLoader(self.test_set, batch_size=1, shuffle=False)
        checkpoint_len = 10000
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.collect_similarity_and_gt, self.progress_report)
        self.decrement_indent()

        self.sample_to_predicted_classes = generate_sample_to_predicted_classes_mapping(self.similarity_gt_list)

    def collect_similarity_and_gt(self, sample_ind, sampled_batch, print_info):
        # Calculate similarity with each text class
        sample_embedding = self.embedding_mat[sample_ind, :]
        similarity_with_classes = {
            x: self.im_txt_similarity_func(sample_embedding, self.class_ind_to_embedding[x])
            for x in self.class_mapping.keys()
        }

        ''' Ground-truth classes and bboxes are not part of the dataset, because these are lists of varying
            length and thus cannot be batched using pytorch's data loader. So we need to extract these from an
            external file. '''
        image_id = sampled_batch['image_id'][0].item()
        gt_classes = self.gt_classes_data[image_id]

        for class_ind, similarity in similarity_with_classes.items():
            self.similarity_gt_list.append((similarity.item(), class_ind in gt_classes, sample_ind, class_ind))

    def predict_classes(self, sample_ind):
        return self.sample_to_predicted_classes[sample_ind]

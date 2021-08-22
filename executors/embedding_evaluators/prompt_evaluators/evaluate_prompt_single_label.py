from executors.embedding_evaluators.prompt_evaluators.evaluate_prompt import PromptEvaluator


class PromptSingleLabelEvaluator(PromptEvaluator):
    """ Evaluate prompt for single-label datasets. """

    def __init__(self, test_set, class_mapping, model_type, model_str, indent):
        super(PromptSingleLabelEvaluator, self).__init__(test_set, class_mapping, model_type, model_str, indent)

        self.multi_label = False

    def predict_classes(self, sample_ind):
        sample_embedding = self.embedding_mat[sample_ind, :]
        similarity_with_classes = {
            x: self.im_txt_similarity_func(sample_embedding, self.class_ind_to_embedding[x])
            for x in self.class_mapping.keys()
        }
        predicted_class = max(similarity_with_classes, key=similarity_with_classes.get)

        return [predicted_class]

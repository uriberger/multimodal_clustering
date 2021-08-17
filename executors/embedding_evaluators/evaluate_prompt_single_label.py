from executors.embedding_evaluators.evaluate_prompt import PromptEvaluator


class PromptSingleLabelEvaluator(PromptEvaluator):
    """ Evaluate prompt for single-label datasets. """

    def predict_classes(self, sample_ind):
        sample_embedding = self.embedding_mat[sample_ind, :]
        similarity_with_classes = {
            x: self.im_txt_similarity_func(sample_embedding, self.class_ind_to_embedding[x])
            for x in self.class_mapping.keys()
        }
        predicted_class = max(similarity_with_classes, key=similarity_with_classes.get)

        return [predicted_class]

    def get_labels_from_batch(self, batch):
        return [batch['label'].to(self.device).item()]

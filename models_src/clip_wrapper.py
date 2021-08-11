from models_src.embedding_model_wrapper import EmbeddingModelWrapper
import clip


class ClipWrapper(EmbeddingModelWrapper):
    """ Wrapper for the Clip model presented in Radford et al. 2021. """

    def __init__(self, device, config, model_dir, indent):
        super(ClipWrapper, self).__init__(device, config, model_dir, indent, 'clip')

    def generate_model(self):
        model, _ = clip.load(self.config.visual_model, self.device)
        return model

    def inference(self, inputs):
        output = self.model.encode_image(inputs)
        self.cached_output = output
        return output

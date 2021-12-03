class ModelConfig:
    def __init__(self,
                 visual_model='resnet18',
                 pretrained_visual_base_model=True,
                 visual_model_path=None,
                 freeze_parameters=False,
                 visual_threshold=0.5,
                 visual_learning_rate=1e-4,
                 text_model='counts_generative',
                 text_threshold=0.5,
                 textual_learning_rate=1e-4,
                 word_embed_dim=300,
                 cluster_num=80
                 ):
        self.visual_model = visual_model
        self.pretrained_visual_base_model = pretrained_visual_base_model
        self.visual_model_path = visual_model_path
        self.freeze_parameters = freeze_parameters
        self.visual_threshold = visual_threshold
        self.visual_learning_rate = visual_learning_rate
        self.text_model = text_model
        self.text_threshold = text_threshold
        self.textual_learning_rate = textual_learning_rate
        self.word_embed_dim = word_embed_dim
        self.cluster_num = cluster_num

    def __str__(self):
        return 'Configuration: ' + str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

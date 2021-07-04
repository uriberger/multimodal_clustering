wanted_image_size = (224, 224)
models_dir = '../models'


class ModelConfig:
    def __init__(self,
                 visual_model='resnet18',
                 pretrained_visual_base_model=True,
                 object_threshold=0.5,
                 visual_learning_rate=1e-4,
                 text_model='counts_generative',
                 noun_threshold=0.5,
                 concept_num=80
                 ):
        self.visual_model = visual_model
        self.pretrained_visual_base_model = pretrained_visual_base_model
        self.object_threshold = object_threshold
        self.visual_learning_rate = visual_learning_rate
        self.text_model = text_model
        self.noun_threshold = noun_threshold
        self.concept_num = concept_num

    def __str__(self):
        return 'Configuration: ' + str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__

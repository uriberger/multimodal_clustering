wanted_image_size = (224, 224)


class Config:
    def __init__(self,
                 image_model='CAMNet',
                 pretrained_image_base_model=True,
                 object_threshold=0.5,
                 image_learning_rate=1e-4,
                 text_model_mode='discriminative',
                 noun_threshold=0.5,
                 concept_num=80
                 ):
        self.image_model = image_model
        self.pretrained_image_base_model = pretrained_image_base_model
        self.object_threshold = object_threshold
        self.image_learning_rate = image_learning_rate
        self.text_model_mode = text_model_mode
        self.noun_threshold = noun_threshold
        self.concept_num = concept_num

    def __str__(self):
        return \
            'Configuration: ' + 'Image model: ' + str(self.image_model) + \
            ', base model pretrained: ' + str(self.pretrained_image_base_model) + \
            ', object threshold: ' + str(self.object_threshold) + \
            ', learning rate: ' + str(self.image_learning_rate) + \
            ', text model mode: ' + str(self.text_model_mode) + \
            ', noun threshold: ' + str(self.noun_threshold) + \
            ', concept num: ' + str(self.concept_num)

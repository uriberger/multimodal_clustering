class Config:
    def __init__(self,
                 image_model='CAMNet',
                 pretrained_image_base_model=True,
                 object_threshold=0.5,
                 lambda_diversity_loss=2,
                 image_learning_rate=1e-4,
                 text_model_mode='discriminative',
                 noun_threshold=0.155
                 ):
        self.image_model = image_model
        self.pretrained_image_base_model = pretrained_image_base_model
        self.object_threshold = object_threshold
        self.lambda_diversity_loss = lambda_diversity_loss
        self.image_learning_rate = image_learning_rate
        self.text_model_mode = text_model_mode
        self.noun_threshold = noun_threshold

    def __str__(self):
        return \
            'Configuration: ' + 'Image model: ' + str(self.image_model) + \
            ', base model pretrained: ' + str(self.pretrained_image_base_model) + \
            ', object threshold: ' + str(self.object_threshold) + \
            ', lambda: ' + str(self.lambda_diversity_loss) + \
            ', learning rate: ' + str(self.image_learning_rate) + \
            ', text model mode: ' + str(self.text_model_mode) + \
            ', noun threshold: ' + str(self.noun_threshold)

import torch
from executors.trainer import Trainer
import torchvision.models as models
import torch.nn as nn


class InstanceDiscriminationTrainer(Trainer):

    def __init__(self, training_set, epoch_num, indent):
        super().__init__(training_set, epoch_num, 100, indent)

        self.model = models.resnet18(pretrained=False).to(self.device)
        instance_num = len(training_set)
        self.model.fc = nn.Linear(in_features=512, out_features=instance_num)

        self.epoch_num = epoch_num
        self.training_set = training_set

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        class_ind = sampled_batch['index']

        self.optimizer.zero_grad()
        output = self.model(image_tensor)
        loss = self.criterion(output, class_ind)
        loss_val = loss.item()

        if print_info:
            self.log_print('Loss: ' + str(loss_val))

        loss.backward()
        self.optimizer.step()

    def pre_training(self):
        return

    def post_training(self):
        torch.save(self.model.state_dict(), 'image_embedder')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class SimCLRModel(nn.Module):
    def __init__(self, output_encoder=True, feature_dim=128):
        super(SimCLRModel, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            # if name == 'conv1':
            #     module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        self.output_encoder = output_encoder

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        if self.output_encoder:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        else:
            return F.normalize(out, dim=-1)


def clean_state_dict(messy_state_dict):
    """ Clean model from unnecessary parameters, created by the thop package in the original SimCLR repo. """
    cleaned_state_dict = {}
    for n, p in messy_state_dict.items():
        if "total_ops" not in n and "total_params" not in n:
            cleaned_state_dict[n] = p

    return cleaned_state_dict


def adjust_projection_in_state_dict(state_dict, output_size):
    """ Change size of projection layer in an existing state dict. """
    new_state_dict = {}
    for n, p in state_dict.items():
        if not n.startswith('g.'):
            new_state_dict[n] = p

    dummy_layer = nn.Linear(2048, output_size)
    new_state_dict['g.weight'] = dummy_layer.weight
    new_state_dict['g.bias'] = dummy_layer.bias

    return new_state_dict

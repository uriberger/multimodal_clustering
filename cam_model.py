import torch
import torch.nn as nn
import torchvision
from torchcam.cams import CAM


class CAMNet(nn.Module):
    def __init__(self, class_num, pretrained_raw_net=False):
        super().__init__()
        self.class_num = class_num

        raw_net = torchvision.models.vgg16_bn(pretrained=pretrained_raw_net)
        self.features = nn.Sequential(*list(raw_net.features.children())[:-10])

        feature_map_num = 1024
        self.last_conv = nn.Conv2d(in_channels=512, out_channels=feature_map_num,
                                   kernel_size=3, stride=1, padding=1)

        self.projector = nn.Linear(in_features=feature_map_num,
                                   out_features=class_num, bias=False)

        self.cam_extractor = CAM(self, target_layer='last_conv', fc_layer='projector')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')


    def forward(self, images_tensor):
        # First, calculate feature maps
        temp_feature_maps = self.features(images_tensor)
        feature_maps = self.last_conv(temp_feature_maps)

        # Next, apply global average pooling
        feature_map_num = feature_maps.shape[1]
        feature_maps = feature_maps.view(feature_maps.shape[0], feature_map_num, -1)
        pooled_vector = feature_maps.mean(dim=2)

        # Finally, project to the wanted dimension
        output = self.projector(pooled_vector)

        return output

    def extract_cam(self, class_ind, output):
        activation_map = self.cam_extractor(class_ind, output)

        return activation_map

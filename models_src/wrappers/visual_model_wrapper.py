###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from models_src.wrappers.cluster_model_wrapper import ClusterModelWrapper
import torch
from torchcam.cams import CAM
from utils.visual_utils import plot_heatmap, generate_visual_model, unnormalize_trans,\
    predict_bboxes_with_activation_maps
import matplotlib.pyplot as plt


class VisualModelWrapper(ClusterModelWrapper):
    """ This class wraps the visual underlying model.
        Unlike the text model wrapper, there's no specific functionality for the visual wrapper. """

    def __init__(self, device, config, model_dir, model_name, indent):
        super().__init__(device, config, model_dir, model_name, indent)
        self.underlying_model.to(self.device)

        if config is not None:
            # This is a new model
            if config.freeze_visual_parameters:
                # Need to freeze all parameters except the last layer
                for param in self.underlying_model.parameters():
                    param.requires_grad = False
                last_layer = list(self.underlying_model.modules())[-1]
                last_layer.weight.requires_grad = True
                last_layer.bias.requires_grad = True

        self.generate_cam_extractor()

        learning_rate = self.config.visual_learning_rate
        self.optimizer = torch.optim.Adam(self.underlying_model.parameters(), lr=learning_rate)

    # Methods inherited from ancestor

    def generate_underlying_model(self):
        return generate_visual_model(self.config.visual_underlying_model, self.config.cluster_num,
                                     self.config.pretrained_visual_underlying_model)

    def training_step(self, inputs, labels):
        loss = self.criterion(self.cached_output, labels)
        loss_val = loss.item()
        self.cached_loss = loss_val

        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()

    def inference(self, inputs):
        output = self.underlying_model(inputs)
        self.cached_output = output
        return output

    def eval(self):
        self.underlying_model.eval()

    def dump_underlying_model(self):
        torch.save(self.underlying_model.state_dict(), self.get_underlying_model_path())

    def load_underlying_model(self):
        self.underlying_model.load_state_dict(torch.load(self.get_underlying_model_path(), map_location=torch.device(self.device)))

    def get_threshold(self):
        return self.config.visual_threshold

    def get_name(self):
        return 'visual'

    # Current class specific functionality

    def generate_cam_extractor(self):
        if self.config.visual_underlying_model == 'resnet18':
            self.cam_extractor = CAM(self.underlying_model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_underlying_model == 'resnet34':
            self.cam_extractor = CAM(self.underlying_model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_underlying_model == 'resnet50':
            self.cam_extractor = CAM(self.underlying_model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_underlying_model == 'resnet101':
            self.cam_extractor = CAM(self.underlying_model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_underlying_model == 'vgg16':
            self.cam_extractor = CAM(self.underlying_model, target_layer='features', fc_layer='classifier')
        elif self.config.visual_underlying_model == 'googlenet':
            self.cam_extractor = CAM(self.underlying_model, target_layer='inception5b', fc_layer='fc')
        elif self.config.visual_underlying_model == 'simclr':
            self.cam_extractor = CAM(self.underlying_model, target_layer='f.7', fc_layer='g')

    """ Extract the class activation mapping for the most recent inference executed,
        for a given cluster. """

    def extract_cam(self, cluster_ind):
        activation_map = self.cam_extractor(cluster_ind, self.cached_output)

        return activation_map

    """ For each of the samples in the recent inference, predict the cluster with the highest probability. """

    def predict_most_probable_clusters(self):
        return torch.max(self.cached_output, dim=1).indices.tolist()

    """ Given an input image tensor, run inference and predict the class activation maps for each of the samples.
        If the input image tensor is None, predict on the last image we inferred on. """

    def predict_activation_maps(self, image_tensor=None):
        if image_tensor is not None:
            # We need to run inference on each image apart, because the cam extraction demands a single hooked tensor
            old_cached_output = self.cached_output  # We don't want to change the cache

        batch_size = image_tensor.shape[0]
        activation_maps = []
        for sample_ind in range(batch_size):
            activation_maps.append([])
            if image_tensor is not None:
                self.inference(image_tensor[[sample_ind], :, :, :])
            predicted_cluster_list = self.predict_cluster_lists()[0]
            for predicted_cluster in predicted_cluster_list:
                activation_map = self.extract_cam(predicted_cluster)
                activation_maps[-1].append((predicted_cluster, activation_map))

        if image_tensor is not None:
            self.cached_output = old_cached_output

        return activation_maps

    """ Given an input image tensor, predict bounding boxes for each of the samples.
        If the input image tensor is None, predict on the last image we inferred on. """

    def predict_bboxes(self, image_tensor=None):
        activation_maps = self.predict_activation_maps(image_tensor)
        predicted_bboxes = predict_bboxes_with_activation_maps(activation_maps)

        return predicted_bboxes

    """ Given an input image tensor, run inference and plot the heatmap for each sample. """

    def demonstrate_heatmap(self, image_tensor, cluster_to_str):
        activation_maps = self.predict_activation_maps(image_tensor)
        batch_size = len(activation_maps)

        for sample_ind in range(batch_size):
            for cluster_ind, activation_map in activation_maps[sample_ind]:
                class_str = ' (associated classes: '
                if cluster_ind in cluster_to_str:
                    class_str += str(cluster_to_str[cluster_ind])
                else:
                    class_str += 'None'
                class_str += ')'
                unnormalized_image_tensor = unnormalize_trans(image_tensor)
                image_obj = plot_heatmap(unnormalized_image_tensor, activation_map, False)
                plt.imshow(image_obj)

                title = 'Heatmap for cluster ' + str(cluster_ind) + '\n' + class_str
                plt.title(title)

                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                plt.show()

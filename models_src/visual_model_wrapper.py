from models_src.unimodal_model_wrapper import UnimodalModelWrapper
import torch
import torch.nn as nn
from torchcam.cams import CAM
from utils.visual_utils import predict_bbox, plot_heatmap, generate_visual_model
import matplotlib.pyplot as plt


class VisualModelWrapper(UnimodalModelWrapper):

    def __init__(self, device, config, model_dir, indent, name=None):
        if name is None:
            name = 'visual'

        super().__init__(device, config, model_dir, indent, name)
        self.model.to(self.device)

        self.generate_cam_extractor()

        self.criterion = nn.BCEWithLogitsLoss()

        learning_rate = self.config.visual_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.cached_loss = None

    def generate_model(self):
        return generate_visual_model(self.config.visual_model, self.config.concept_num,
                                     self.config.pretrained_visual_base_model)

    def generate_cam_extractor(self):
        if self.config.visual_model == 'resnet18':
            self.cam_extractor = CAM(self.model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_model == 'resnet34':
            self.cam_extractor = CAM(self.model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_model == 'resnet101':
            self.cam_extractor = CAM(self.model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_model == 'vgg16':
            self.cam_extractor = CAM(self.model, target_layer='features', fc_layer='classifier')
        elif self.config.visual_model == 'googlenet':
            self.cam_extractor = CAM(self.model, target_layer='inception5b', fc_layer='fc')

    def training_step(self, inputs, labels):
        loss = self.criterion(self.cached_output, labels)
        loss_val = loss.item()
        self.cached_loss = loss_val

        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()

    def inference(self, inputs):
        output = self.model(inputs)
        self.cached_output = output
        return output

    def print_info_on_inference(self):
        output = self.cached_output

        batch_size = output.shape[0]
        concept_num = self.config.concept_num

        best_winner = torch.max(torch.tensor(
            [len([i for i in range(batch_size) if torch.argmax(output[i, :]).item() == j])
             for j in range(concept_num)])).item()
        return 'Best winner won ' + str(best_winner) + ' times out of ' + str(batch_size)

    def print_info_on_loss(self):
        return 'Loss: ' + str(self.cached_loss)

    def eval(self):
        self.model.eval()

    def dump_model(self):
        torch.save(self.model.state_dict(), self.get_model_path())

    def load_model(self):
        self.model.load_state_dict(torch.load(self.get_model_path(), map_location=torch.device(self.device)))

    def extract_cam(self, class_ind):
        activation_map = self.cam_extractor(class_ind, self.cached_output)

        return activation_map

    def predict_concept_indicators(self):
        with torch.no_grad():
            prob_output = torch.sigmoid(self.cached_output)
            concepts_indicator = torch.zeros(prob_output.shape).to(self.device)
            concepts_indicator[prob_output > self.config.object_threshold] = 1

        return concepts_indicator

    def predict_activation_maps(self, image_tensor):
        # We need to run inference on each image apart, because the cam extraction demands a single hooked tensor
        old_cached_output = self.cached_output  # We don't want to change the cache

        batch_size = image_tensor.shape[0]
        activation_maps = []
        for sample_ind in range(batch_size):
            activation_maps.append([])
            self.inference(image_tensor[[sample_ind], :, :, :])
            predicted_class_list = self.predict_concept_lists()[0]
            for predicted_class in predicted_class_list:
                activation_map = self.extract_cam(predicted_class)
                activation_maps[-1].append(activation_map)

        self.cached_output = old_cached_output

        return activation_maps

    def predict_bboxes(self, image_tensor):
        activation_maps = self.predict_activation_maps(image_tensor)
        predicted_bboxes = self.predict_bboxes_with_activation_maps(activation_maps)

        return predicted_bboxes

    def predict_bboxes_with_activation_maps(self, activation_maps):
        predicted_bboxes = []
        for sample_activation_maps in activation_maps:
            predicted_bboxes.append([])
            for predicted_class_activation_map in sample_activation_maps:
                bbox = predict_bbox(predicted_class_activation_map)
                predicted_bboxes[-1].append(bbox)

        return predicted_bboxes

    def plot_heatmap(self, image_tensor, concept_to_str):
        old_cached_output = self.cached_output  # We don't want to change the cache

        batch_size = image_tensor.shape[0]
        for sample_ind in range(batch_size):
            self.inference(image_tensor[[sample_ind], :, :, :])
            predicted_class_list = self.predict_concept_lists()[0]
            for predicted_class in predicted_class_list:
                if predicted_class in concept_to_str:
                    class_str = str(concept_to_str[predicted_class])
                else:
                    class_str = '[]'
                activation_map = self.extract_cam(predicted_class)
                image_obj = plot_heatmap(image_tensor, activation_map, False)
                plt.imshow(image_obj)
                plt.title('Heatmap for class ' + str(predicted_class) +
                          class_str)
                plt.show()

        self.cached_output = old_cached_output

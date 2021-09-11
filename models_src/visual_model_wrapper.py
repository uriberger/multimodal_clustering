from models_src.unimodal_model_wrapper import UnimodalModelWrapper
import torch
from torchcam.cams import CAM
from utils.visual_utils import predict_bbox, plot_heatmap, generate_visual_model, unnormalize_trans
import matplotlib.pyplot as plt
from models_src.simclr import clean_state_dict, adjust_projection_in_state_dict


class VisualModelWrapper(UnimodalModelWrapper):

    def __init__(self, device, config, model_dir, model_name, indent):
        super().__init__(device, config, model_dir, model_name, indent)
        self.model.to(self.device)

        if config is not None:
            if config.visual_model_path:
                orig_state_dict = torch.load(config.visual_model_path, map_location=self.device)
                cleaned_state_dict = clean_state_dict(orig_state_dict)
                adjusted_state_dict = adjust_projection_in_state_dict(cleaned_state_dict, config.concept_num)
                self.model.load_state_dict(adjusted_state_dict)

            if config.freeze_parameters:
                for param in self.model.parameters():
                    param.requires_grad = False
                last_layer = list(self.model.modules())[-1]
                last_layer.weight.requires_grad = True
                last_layer.bias.requires_grad = True

        self.generate_cam_extractor()

        learning_rate = self.config.visual_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def generate_model(self):
        return generate_visual_model(self.config.visual_model, self.config.concept_num,
                                     self.config.pretrained_visual_base_model)

    def generate_cam_extractor(self):
        if self.config.visual_model == 'resnet18':
            self.cam_extractor = CAM(self.model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_model == 'resnet34':
            self.cam_extractor = CAM(self.model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_model == 'resnet50':
            self.cam_extractor = CAM(self.model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_model == 'resnet101':
            self.cam_extractor = CAM(self.model, target_layer='layer4', fc_layer='fc')
        elif self.config.visual_model == 'vgg16':
            self.cam_extractor = CAM(self.model, target_layer='features', fc_layer='classifier')
        elif self.config.visual_model == 'googlenet':
            self.cam_extractor = CAM(self.model, target_layer='inception5b', fc_layer='fc')
        elif self.config.visual_model == 'simclr':
            self.cam_extractor = CAM(self.model, target_layer='f.7', fc_layer='g')

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

    def predict_classes(self):
        return torch.max(self.cached_output, dim=1).indices.tolist()

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
                unnormalized_image_tensor = unnormalize_trans(image_tensor)
                image_obj = plot_heatmap(unnormalized_image_tensor, activation_map, False)
                plt.imshow(image_obj)
                plt.title('Heatmap for class ' + str(predicted_class) +
                          class_str)
                plt.show()

        self.cached_output = old_cached_output

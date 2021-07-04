from models_src.model_wrapper import ModelWrapper
import torch
import torch.nn as nn
import torchvision.models as models
from torchcam.cams import CAM
from utils.visual_utils import predict_bbox
from torchvision.transforms.functional import to_pil_image
from models_src.model_config import wanted_image_size
from PIL import ImageDraw
import matplotlib.pyplot as plt


def generate_visual_model(model_str, concept_num, pretrained_base):
    if model_str == 'resnet18':
        model = models.resnet18(pretrained=pretrained_base)
        model.fc = nn.Linear(512, concept_num)
    elif model_str == 'vgg16':
        model = models.vgg16(pretrained=pretrained_base)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Linear(512, concept_num)
    elif model_str == 'googlenet':
        model = models.googlenet(pretrained=pretrained_base, aux_logits=False)
        model.fc = nn.Linear(1024, concept_num)

    return model


class VisualModelWrapper(ModelWrapper):

    def __init__(self, device, config, model_dir, indent, name=None):
        if name is None:
            name = 'visual'

        super().__init__(device, config, model_dir, indent, name)
        self.model = generate_visual_model(config.visual_model, config.concept_num,
                                           config.pretrained_visual_base_model)
        self.model.to(self.device)
        self.load_model_if_needed()

        self.generate_cam_extractor()

        self.criterion = nn.BCEWithLogitsLoss()

        learning_rate = config.visual_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.cached_loss = None

    def generate_cam_extractor(self):
        if self.config.visual_model == 'resnet18':
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

    def no_grad(self):
        self.model.eval()

    def dump(self):
        self.dump_config()
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

    def predict_bboxes(self, image_tensor):
        # We need to run inference on each image apart, because the cam extraction demands a single hooked tensor
        batch_size = image_tensor.shape[0]
        predicted_bboxes = []
        for sample_ind in range(batch_size):
            predicted_bboxes.append([])
            self.inference(image_tensor[[sample_ind], :, :, :])
            predicted_class_list = self.predict_concept_lists()[0]
            for predicted_class in predicted_class_list:
                activation_map = self.extract_cam(predicted_class)
                bbox = predict_bbox(activation_map)
                predicted_bboxes[-1].append(bbox)
        return predicted_bboxes

    def predict_and_draw_bounding_boxes(self, image_tensor, gt_bboxes=None):
        self.inference(image_tensor)
        predicted_concepts = self.predict_concept_indicators()
        image_obj = to_pil_image(image_tensor.view(3, wanted_image_size[0], wanted_image_size[1]))
        draw = ImageDraw.Draw(image_obj)

        # Draw predicted boxes
        for concept in predicted_concepts:
            activation_map = self.extract_cam(concept)
            bbox = predict_bbox(activation_map)
            draw.rectangle(bbox, outline=(255, 0, 0))

        # Draw gt boxes
        for bbox in gt_bboxes:
            draw.rectangle(bbox, outline=(0, 255, 0))

        plt.imshow(image_obj)
        plt.show()

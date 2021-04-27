from cam_model import CAMNet, predict_classes
from train_cam_from_golden import predict_bbox
import torchvision.models as models
from noun_identifier import NounIdentifier, preprocess_token
import torch.utils.data as data
import time
from aux_functions import log_print, calc_ious
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchcam.cams import CAM
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
import spacy
from golden_semantic_class_dataset import noun_tags
from coco import generate_bboxes_dataset_coco
from config import wanted_image_size


def generate_model(model_str, class_num, pretrained_base):
    if model_str == 'CAMNet':
        model = CAMNet(class_num, pretrained_raw_net=pretrained_base)
    elif model_str == 'resnet18':
        model = models.resnet18(pretrained=pretrained_base)
        model.fc = nn.Linear(512, class_num)
    elif model_str == 'vgg16':
        model = models.vgg16(pretrained=pretrained_base)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1],
                                               nn.Linear(4096, class_num))
        model.classifier = nn.Linear(25088, class_num)

    return model


def generate_cam_extractor(image_model, image_model_str):
    if image_model_str == 'CAMNet':  # TODO fix the content of this if statement
        cam_extractor = CAM(image_model, target_layer='layer4', fc_layer='fc')
    elif image_model_str == 'resnet18':
        cam_extractor = CAM(image_model, target_layer='layer4', fc_layer='fc')
    elif image_model_str == 'vgg16':  # TODO fix the content of this if statement
        cam_extractor = CAM(image_model, target_layer='layer4', fc_layer='fc')

    return cam_extractor


def loss_with_weight_constraint(output, labels, fc_layer_weights, lambda_diversity_loss):
    bce_loss = nn.BCEWithLogitsLoss()(output, labels)

    norm_vec = torch.sum(fc_layer_weights, dim=1)
    diversity_loss = torch.max(norm_vec) - torch.min(norm_vec)

    total_loss = bce_loss + lambda_diversity_loss * diversity_loss

    return bce_loss, diversity_loss, total_loss


def train_joint_model(timestamp, training_set, epoch_num, config):
    function_name = 'train_joint_model'
    indent = 1

    image_criterion = loss_with_weight_constraint
    losses = []

    # Configuration
    noun_threshold = config.noun_threshold
    object_threshold = config.object_threshold
    lambda_diversity_loss = config.lambda_diversity_loss
    pretrained_base = config.pretrained_image_base_model
    learning_rate = config.image_learning_rate
    text_model_mode = config.text_model_mode
    image_model_str = config.image_model
    class_num = config.class_num

    image_model = generate_model(image_model_str, class_num, pretrained_base)
    text_model = NounIdentifier(class_num, text_model_mode)

    image_optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    image_model.to(device)

    for epoch_ind in range(epoch_num):
        log_print(function_name, indent + 1, 'Starting epoch ' + str(epoch_ind))

        dataloader = data.DataLoader(training_set, batch_size=100, shuffle=True)
        # dataloader = data.DataLoader(training_set, batch_size=5, shuffle=True)
        checkpoint_len = 50
        checkpoint_time = time.time()
        for i_batch, sampled_batch in enumerate(dataloader):
            if i_batch % checkpoint_len == 0:
                print_info = True
            else:
                print_info = False

            if print_info:
                log_print(function_name, indent+2, 'Starting batch ' + str(i_batch) +
                          ' out of ' + str(len(dataloader)) +
                          ', time from previous checkpoint ' + str(time.time() - checkpoint_time))
                checkpoint_time = time.time()

            image_tensor = sampled_batch['image'].to(device)
            captions = sampled_batch['caption']
            batch_size = len(captions)
            token_lists = []
            for caption in captions:
                token_list = caption.split()
                token_list = [preprocess_token(token) for token in token_list]
                token_lists.append(token_list)

            image_optimizer.zero_grad()
            image_output = image_model(image_tensor)

            if print_info:
                best_winner = torch.max(torch.tensor(
                    [len([i for i in range(batch_size) if torch.argmax(image_output[i, :]).item() == j])
                     for j in range(class_num)])).item()
                log_print(function_name, indent+3, 'Best winner won ' + str(best_winner) + ' times out of ' + str(batch_size))

            # Train text model, assuming that the image model is already trained
            with torch.no_grad():
                predicted_classes_by_image_list = predict_classes(image_output, confidence_threshold=object_threshold)
            if print_info:
                predictions_num = sum([len(predicted_classes_by_image_list[i]) for i in range(batch_size)])
                log_print(function_name, indent + 3, 'Predicted ' + str(predictions_num) + ' classes according to image')
            for caption_ind in range(batch_size):
                predicted_classes_by_image = predicted_classes_by_image_list[caption_ind]
                for token in token_lists[caption_ind]:
                    for semantic_class_ind in predicted_classes_by_image:
                        text_model.document_co_occurrence(token, semantic_class_ind)

            # Train image model, assuming that the text model is already trained
            with torch.no_grad():
                text_model.calculate_probs()
                label_tensor = torch.zeros(batch_size, class_num).to(device)
                for caption_ind in range(batch_size):
                    predicted_class_list = []
                    for token in token_lists[caption_ind]:
                        prediction_res = text_model.predict_semantic_class(token)
                        if prediction_res is None:
                            # Never seen this token before
                            continue
                        predicted_class, prob = prediction_res
                        if prob >= noun_threshold:
                            predicted_class_list.append(predicted_class)
                    label_tensor[caption_ind, torch.tensor(predicted_class_list).long()] = 1.0

            predictions_num = torch.sum(label_tensor).item()
            if print_info:
                log_print(function_name, indent + 3, 'Predicted ' + str(predictions_num) + ' classes according to text')
            # Train image model
            bce_loss, diversity_loss, loss =\
                image_criterion(image_output, label_tensor,
                                image_model.fc.weight, lambda_diversity_loss)
            bce_loss_val = bce_loss.item()
            weight_loss_val = diversity_loss.item()
            if print_info:
                log_print(function_name, indent + 3, 'BCE loss: ' + str(bce_loss_val) +
                          ' diversity loss: ' + str(weight_loss_val))
            losses.append(bce_loss_val)

            loss.backward()
            image_optimizer.step()
        image_model_path = os.path.join(timestamp, 'image_model.mdl')
        torch.save(image_model.state_dict(), image_model_path)
        text_model_path = os.path.join(timestamp, 'text_model.mdl')
        torch.save(text_model, text_model_path)


def test_models(timestamp, test_set, config):
    function_name = 'test_models'
    indent = 1

    # Config parameters
    object_threshold = config.object_threshold
    noun_threshold = config.noun_threshold
    pretrained_base = config.pretrained_image_base_model
    image_model_str = config.image_model
    class_num = config.class_num

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    _, img_bboxes_val_set, _ = generate_bboxes_dataset_coco()

    # Load models
    image_model = generate_model(image_model_str, class_num, pretrained_base)
    # image_model_path = os.path.join(timestamp, 'image_model.mdl')
    image_model_path = 'non_pretrained_image_model.mdl'
    image_model.load_state_dict(torch.load(image_model_path, map_location=torch.device(device)))
    image_model.eval()

    # text_model_path = os.path.join(timestamp, 'text_model.mdl')
    text_model_path = 'non_pretrained_text_model.mdl'
    text_model = torch.load(text_model_path)
    nlp = spacy.load("en_core_web_sm")

    image_tp = 0
    image_fp = 0
    image_fn = 0
    text_tp = 0
    text_tn = 0
    text_fp = 0
    text_fn = 0

    dataloader = data.DataLoader(test_set, batch_size=1, shuffle=True)
    checkpoint_len = 1000
    checkpoint_time = time.time()
    for i_batch, sampled_batch in enumerate(dataloader):
        if i_batch % checkpoint_len == 0:
            log_print(function_name, indent + 1, 'Starting batch ' + str(i_batch) +
                      ' out of ' + str(len(dataloader)) +
                      ', time from previous checkpoint ' + str(time.time() - checkpoint_time))
            checkpoint_time = time.time()

        with torch.no_grad():
            # Test image model
            image_tensor = sampled_batch['image'].to(device)
            # draw_bounding_box(image_model, image_model_str, image_tensor)
            image_id = sampled_batch['image_id'].item()
            if image_id in img_bboxes_val_set:
                orig_image_size = sampled_batch['orig_image_size']
                gt_bboxes_with_classes = img_bboxes_val_set[image_id]
                gt_bboxes = [(
                    int((x[0][0] / orig_image_size[0])*wanted_image_size[0]),
                    int((x[0][1] / orig_image_size[1])*wanted_image_size[1]),
                    int((x[0][2] / orig_image_size[0])*wanted_image_size[0]),
                    int((x[0][3] / orig_image_size[1])*wanted_image_size[1])
                ) for x in gt_bboxes_with_classes]
                cur_tp, cur_fp, cur_fn = \
                    evaluate_bounding_boxes(image_model, image_tensor, object_threshold, gt_bboxes, image_model_str)
                image_tp += cur_tp
                image_fp += cur_fp
                image_fn += cur_fn

            # Test text model
            caption = sampled_batch['caption'][0]
            doc = nlp(caption)
            for token in doc:
                token_str = preprocess_token(token.text)
                prediction = text_model.predict_semantic_class(token_str)
                if prediction is None:
                    continue
                is_noun_prediction = prediction[1] >= noun_threshold
                is_noun_gt = token.tag_ in noun_tags
                if is_noun_prediction and is_noun_gt:
                    text_tp += 1
                elif is_noun_prediction and (not is_noun_gt):
                    text_fp += 1
                elif (not is_noun_prediction) and is_noun_gt:
                    text_fn += 1
                else:
                    text_tn += 1

    log_print(function_name, indent,
              'image: tp ' + str(image_tp) +
              ', fp ' + str(image_fp) +
              ', fn ' + str(image_fn))
    image_precision = image_tp/(image_tp+image_fp)
    log_print(function_name, indent, 'Image precision: ' + str(image_precision))
    image_recall = image_tp/(image_tp+image_fn)
    log_print(function_name, indent, 'Image recall: ' + str(image_recall))
    image_f1 = 2*(image_precision*image_recall)/(image_precision+image_recall)
    log_print(function_name, indent, 'Image F1: ' + str(image_f1))

    log_print(function_name, indent,
              'Text: tp ' + str(text_tp) +
              ', tn ' + str(text_tn) +
              ', fp ' + str(text_fp) +
              ', fn ' + str(text_fn))
    text_precision = text_tp / (text_tp + text_fp)
    log_print(function_name, indent, 'Text precision: ' + str(text_precision))
    text_recall = text_tp / (text_tp + text_fn)
    log_print(function_name, indent, 'Text recall: ' + str(text_recall))
    text_f1 = 2 * (text_precision * text_recall) / (text_precision + text_recall)
    log_print(function_name, indent, 'Text F1: ' + str(text_f1))


def report_prediction(image_tensor, predictions):
    image_for_plt = image_tensor.view(3, image_tensor.shape[2], image_tensor.shape[3])
    image_for_plt = image_for_plt.permute(1, 2, 0)
    plt.imshow(image_for_plt)
    plt.show()

    print('Predicted the following classes: ' + str(predictions[0]))


def draw_bounding_box(image_model, image_model_str, image_tensor):
    cam_extractor = generate_cam_extractor(image_model, image_model_str)
    image_output = image_model(image_tensor)
    activation_map = cam_extractor(image_output.squeeze(0).argmax().item(), image_output)
    bbox = predict_bbox(activation_map)
    image_obj = to_pil_image(image_tensor.view(3, 224, 224))
    draw = ImageDraw.Draw(image_obj)
    draw.rectangle(bbox)
    plt.imshow(image_obj)
    plt.show()


def predict_bboxes(image_model, image_tensor, object_threshold, image_model_str):
    cam_extractor = generate_cam_extractor(image_model, image_model_str)
    image_output = image_model(image_tensor)
    predicted_classes = predict_classes(image_output, confidence_threshold=object_threshold)[0]
    predicted_bboxes = []
    for predicted_class in predicted_classes:
        activation_map = cam_extractor(predicted_class, image_output)
        bbox = predict_bbox(activation_map)
        predicted_bboxes.append(bbox)
    return predicted_bboxes


def evaluate_bounding_boxes(image_model, image_tensor, object_threshold, gt_bboxes, image_model_str, iou_threshold=0.5):
    predicted_bboxes = predict_bboxes(image_model, image_tensor, object_threshold, image_model_str)

    # image_obj = to_pil_image(image_tensor.view(3, 224, 224))
    # draw = ImageDraw.Draw(image_obj)
    # for bbox in predicted_bboxes:
    #     draw.rectangle(bbox, outline=(255, 0, 0))
    # for bbox in gt_bboxes:
    #     draw.rectangle(bbox, outline=(0, 255, 0))
    # plt.imshow(image_obj)
    # plt.show()

    gt_bbox_num = len(gt_bboxes)
    gt_bboxes = torch.stack([torch.tensor(x) for x in gt_bboxes])
    if len(predicted_bboxes) > 0:
        predicted_bboxes = torch.stack([torch.tensor(x) for x in predicted_bboxes])
        ious = calc_ious(gt_bboxes, predicted_bboxes)

    tp = 0
    fp = 0
    identifed_gt_inds = {}
    for predicted_bbox_ind in range(len(predicted_bboxes)):
        for gt_bbox_ind in range(gt_bbox_num):
            iou = ious[gt_bbox_ind, predicted_bbox_ind]
            if iou >= iou_threshold:
                tp += 1
                identifed_gt_inds[gt_bbox_ind] = True
                continue
            fp += 1
    fn = gt_bbox_num - len(identifed_gt_inds)

    return tp, fp, fn

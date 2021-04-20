from cam_model import CAMNet, predict_classes
import torchvision.models as models
from noun_identifier import NounIdentifier, preprocess_token
import torch.utils.data as data
import time
from aux_functions import log_print
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


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


def loss_with_weight_constraint(output, labels, fc_layer_weights, lambda_diversity_loss):
    bce_loss = nn.BCEWithLogitsLoss()(output, labels)

    norm_vec = torch.sum(fc_layer_weights, dim=1)
    diversity_loss = torch.max(norm_vec) - torch.min(norm_vec)

    total_loss = bce_loss + lambda_diversity_loss * diversity_loss

    return bce_loss, diversity_loss, total_loss


def train_joint_model(timestamp, training_set, class_num, epoch_num, config):
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
        checkpoint_len = 1
        checkpoint_time = time.time()
        for i_batch, sampled_batch in enumerate(dataloader):
            if i_batch % checkpoint_len == 0:
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

            best_winner = torch.max(torch.tensor(
                [len([i for i in range(batch_size) if torch.argmax(image_output[i, :]).item() == j])
                 for j in range(class_num)])).item()
            best_winner_ind = torch.argmax(torch.tensor(
                [len([i for i in range(batch_size) if torch.argmax(image_output[i, :]).item() == j])
                 for j in range(class_num)])).item()
            best_winner_weight_sum = torch.sum(image_model.fc.weight[best_winner_ind, :]).item()
            weight_mean = sum([torch.sum(image_model.fc.weight[i, :]).item() for i in range(class_num)])/class_num
            log_print(function_name, indent+3, 'Best winner won ' + str(best_winner) + ' times out of ' + str(batch_size))
            log_print(function_name, indent+3, 'Best winner weights: ' + str(best_winner_weight_sum)
                      + ', mean weight: ' + str(weight_mean))

            # Train text model, assuming that the image model is already trained
            with torch.no_grad():
                predicted_classes_by_image_list = predict_classes(image_output, confidence_threshold=object_threshold)
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
            log_print(function_name, indent + 3, 'Predicted ' + str(predictions_num) + ' classes according to text')
            # Train image model
            bce_loss, diversity_loss, loss =\
                image_criterion(image_output, label_tensor,
                                image_model.fc.weight, lambda_diversity_loss)
            bce_loss_val = bce_loss.item()
            weight_loss_val = diversity_loss.item()
            log_print(function_name, indent + 3, 'BCE loss: ' + str(bce_loss_val) +
                      ' diversity loss: ' + str(weight_loss_val))
            losses.append(bce_loss_val)

            loss.backward()
            image_optimizer.step()
        image_model_path = os.path.join(timestamp, 'image_model.mdl')
        torch.save(image_model.state_dict(), image_model_path)
        text_model_path = os.path.join(timestamp, 'text_model.mdl')
        torch.save(text_model, text_model_path)


def test_models(timestamp, test_set, class_num, config):
    function_name = 'test_models'
    indent = 1

    # Config parameters
    object_threshold = config.object_threshold
    pretrained_base = config.pretrained_image_base_model
    image_model_str = config.image_model

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Load models
    image_model = generate_model(image_model_str, class_num, pretrained_base)
    image_model_path = os.path.join(timestamp, 'image_model.mdl')
    image_model.load_state_dict(torch.load(image_model_path, map_location=torch.device(device)))
    image_model.eval()

    text_model_path = os.path.join(timestamp, 'text_model.mdl')
    text_model = torch.load(text_model_path)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    dataloader = data.DataLoader(test_set, batch_size=1, shuffle=True)
    checkpoint_len = 100
    checkpoint_time = time.time()
    for i_batch, sampled_batch in enumerate(dataloader):
        if i_batch % checkpoint_len == 0:
            log_print(function_name, indent + 1, 'Starting batch ' + str(i_batch) +
                      ' out of ' + str(len(dataloader)) +
                      ', time from previous checkpoint ' + str(time.time() - checkpoint_time))
            checkpoint_time = time.time()

        # Test image model
        with torch.no_grad():
            image_tensor = sampled_batch['image'].to(device)
            image_output = image_model(image_tensor)
            predicted_classes_by_image_list = predict_classes(image_output, confidence_threshold=object_threshold)
            # report_prediction(image_tensor, predicted_classes_by_image_list)
            if sampled_batch['gt_classes'][0].item() == 4:
                if 1 in predicted_classes_by_image_list[0]:
                    tp += 1
                else:
                    fn += 1
                if 0 in predicted_classes_by_image_list[0]:
                    fp += 1
                else:
                    tn += 1
            if sampled_batch['gt_classes'][0].item() == 14:
                if 0 in predicted_classes_by_image_list[0]:
                    tp += 1
                else:
                    fn += 1
                if 1 in predicted_classes_by_image_list[0]:
                    fp += 1
                else:
                    tn += 1

    print(tp, tn, fp, fn)
    precision = tp/(tp+fp)
    print('Precision: ' + str(precision))
    recall = tp/(tp+fn)
    print('Recall: ' + str(recall))
    f1 = 2*(precision*recall)/(precision+recall)
    print('F1: ' + str(f1))


def report_prediction(image_tensor, predictions):
    image_for_plt = image_tensor.view(3, image_tensor.shape[2], image_tensor.shape[3])
    image_for_plt = image_for_plt.permute(1, 2, 0)
    plt.imshow(image_for_plt)
    plt.show()

    print('Predicted the following classes: ' + str(predictions[0]))

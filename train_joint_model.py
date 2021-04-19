from cam_model import CAMNet, predict_classes
import torchvision.models as models
from noun_identifier import NounIdentifier, preprocess_token
import torch.utils.data as data
import time
from aux_functions import log_print
import torch
import torch.nn as nn
import os


def loss_with_weight_constraint(output, labels, fc_layer_weights, lambda_diversity_loss):
    bce_loss = nn.BCEWithLogitsLoss()(output, labels)

    # norm_vec = torch.norm(fc_layer_weights, dim=1, p=2)
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

    if image_model_str == 'CAMNet':
        image_model = CAMNet(class_num, pretrained_raw_net=pretrained_base)
    elif image_model_str == 'resnet18':
        image_model = models.resnet18(pretrained=pretrained_base)
        image_model.fc = nn.Linear(512, class_num)
    elif image_model_str == 'vgg16':
        image_model = models.vgg16(pretrained=pretrained_base)
        image_model.classifier = nn.Sequential(*list(image_model.classifier.children())[:-1],
                                               nn.Linear(4096, class_num))
        image_model.classifier = nn.Linear(25088, class_num)

    text_model = NounIdentifier(class_num, text_model_mode)

    ''' Use SGD and not ADAM. After one iteration, ADAM makes the output results much lower,
    so that when I'm predicting the classes according to the image, no one crosses the threshold.'''
    image_optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)

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

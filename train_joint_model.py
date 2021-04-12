from cam_model import CAMNet, predict_classes
from noun_identifier import NounIdentifier, preprocess_token
import torch.utils.data as data
import time
from aux_functions import log_print
import torch
import torch.nn as nn
import numpy as np


def train_joint_model(training_set, class_num, epoch_num):
    function_name = 'train_joint_model'
    indent = 1

    image_criterion = nn.BCEWithLogitsLoss()
    losses = []
    noun_threshold = 0.155
    object_threshold = 0.5

    image_model = CAMNet(class_num, pretrained_raw_net=True)
    text_model = NounIdentifier(class_num)

    ''' Use SGD and not ADAM. After one iteration, ADAM makes the output results much lower,
    so that when I'm predicting the classes according to the image, no one crosses the threshold.'''
    image_optimizer = torch.optim.SGD(image_model.parameters(), lr=1e-4)

    image_model.to(image_model.device)

    dataloader = data.DataLoader(training_set, batch_size=100, shuffle=True)
    checkpoint_len = 100
    checkpoint_time = time.time()
    for i_batch, sampled_batch in enumerate(dataloader):
        if i_batch % checkpoint_len == 0:
            log_print(function_name, indent+1, 'Starting batch ' + str(i_batch) +
                      ' out of ' + str(len(dataloader)) +
                      ', time from previous checkpoint ' + str(time.time() - checkpoint_time))
            checkpoint_time = time.time()

        image_tensor = sampled_batch['image'].to(image_model.device)
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
        log_print(function_name, indent+2, 'Best winner won ' + str(best_winner) + ' times out of ' + str(batch_size))

        # Train text model, assuming that the image model is already trained
        with torch.no_grad():
            predicted_classes_by_image_list = predict_classes(image_output, confidence_threshold=object_threshold)
        predictions_num = sum([len(predicted_classes_by_image_list[i]) for i in range(batch_size)])
        log_print(function_name, indent + 2, 'Predicted ' + str(predictions_num) + ' classes according to image')
        for caption_ind in range(batch_size):
            predicted_classes_by_image = predicted_classes_by_image_list[caption_ind]
            for token in token_lists[caption_ind]:
                for semantic_class_ind in predicted_classes_by_image:
                    text_model.document_co_occurence(token, semantic_class_ind)

        # Train image model, assuming that the text model is already trained
        with torch.no_grad():
            no_prediction_count = 0
            label_tensor = torch.zeros(batch_size, class_num)
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
                # if len(predicted_class_list) == 0:
                #     # We couldn't predict the classes for this caption. Create a random prediction
                #     predicted_class_list.append(np.random.randint(class_num))
                #     predicted_class_list.append(np.random.randint(class_num))
                #     predicted_class_list.append(np.random.randint(class_num))
                #     predicted_class_list.append(np.random.randint(class_num))
                #     predicted_class_list.append(np.random.randint(class_num))
                #     no_prediction_count += 1
                label_tensor[caption_ind, torch.tensor(predicted_class_list).long()] = 1.0
            # print('Couldn\'t predict classes for ' + str(no_prediction_count) + ' captions out of ' + str(batch_size))

        predictions_num = torch.sum(label_tensor).item()
        log_print(function_name, indent + 2, 'Predicted ' + str(predictions_num) + ' classes according to text')
        # Train image model
        loss = image_criterion(image_output, label_tensor)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        image_optimizer.step()

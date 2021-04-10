import torch
from tqdm import trange
from PIL import Image
from coco import get_image_path
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
import torch.utils.data as data
from aux_functions import log_print
import time
import os


def train_epoch(training_set, model, optimizer):
    function_name = 'train_epoch'
    indent = 2
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    dataloader = data.DataLoader(training_set, batch_size=4, shuffle=True)
    checkpoint_len = 100
    checkpoint_time = time.time()
    for i_batch, sampled_batch in enumerate(dataloader):
        if i_batch % checkpoint_len == 0:
            log_print(function_name, indent+1, 'Starting batch ' + str(i_batch) +
                      ' out of ' + str(len(dataloader)) +
                      ', time from previous checkpoint ' + str(time.time() - checkpoint_time))
            checkpoint_time = time.time()

        image_tensor = sampled_batch['image'].to(model.device)
        label_tensor = sampled_batch['label'].to(model.device)

        output = model(image_tensor)
        loss = criterion(output, label_tensor)
        loss_val = loss.item()
        losses.append(loss_val)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    log_print(function_name, indent, 'Avg loss = ' + str(avg_loss))


def advance_pixel(row, col, h, w):
    if col < w - 1:
        return row, col + 1
    else:
        if row < h - 1:
            return row + 1, 0
        else:
            return None, None

def get_final_cc(cc_ind, cc_to_final_cc):
    while cc_ind in cc_to_final_cc:
        cc_ind = cc_to_final_cc[cc_ind]
    return cc_ind


def find_neighbors_cc_ind(pixel_to_cc_ind, cc_to_final_cc, width, pixel_ind):
    # Need the check the neighbors we already visited: upper-left, upper, upper-right, left
    cc_inds = []
    if pixel_ind >= width:
        # This is not the first row
        if pixel_ind % width > 0:
            # This is not the first column, check upper-left neighbor
            upper_left_ind = pixel_ind - width - 1
            if upper_left_ind in pixel_to_cc_ind:
                cc_inds.append(get_final_cc(pixel_to_cc_ind[upper_left_ind], cc_to_final_cc))
        # Check upper neighbor
        upper_ind = pixel_ind - width
        if upper_ind in pixel_to_cc_ind:
            cc_inds.append(get_final_cc(pixel_to_cc_ind[upper_ind], cc_to_final_cc))

        if pixel_ind % width < width-1:
            # This is not the last column, check upper-right neighbor
            upper_right_ind = pixel_ind - width + 1
            if upper_right_ind in pixel_to_cc_ind:
                cc_inds.append(get_final_cc(pixel_to_cc_ind[upper_right_ind], cc_to_final_cc))

    if pixel_ind % width > 0:
        # This is not the first column, check left neighbor
        left_ind = pixel_ind - 1
        if left_ind in pixel_to_cc_ind:
            cc_inds.append(get_final_cc(pixel_to_cc_ind[left_ind], cc_to_final_cc))

    # If multiple cc's are found, update the cc mapping
    cc_inds = list(set(cc_inds))
    if len(cc_inds) > 0:
        cc_ind = cc_inds[0]
        for other_cc_ind in cc_inds[1:]:
            cc_to_final_cc[other_cc_ind] = cc_ind
    else:
        cc_ind = None

    return cc_ind


def create_final_cc_mapping(cc_num, cc_to_final_cc):
    return {x: get_final_cc(x, cc_to_final_cc) for x in range(cc_num)}


def argmax_cc(pixel_to_cc_ind):
    cc_ind_to_size = {}
    for _, cc_ind in pixel_to_cc_ind.items():
        if cc_ind not in cc_ind_to_size:
            cc_ind_to_size[cc_ind] = 0
        cc_ind_to_size[cc_ind] += 1
    largest_cc_size = 0
    largest_cc_ind = 0
    for cc_ind, cc_size in cc_ind_to_size.items():
        if cc_size > largest_cc_size:
            largest_cc_size = cc_size
            largest_cc_ind = cc_ind

    return largest_cc_ind


def map_pixel_to_cc(input_tensor):
    next_cc_ind = 0
    pixel_to_cc_ind = {}
    cc_to_final_cc = {}
    height = input_tensor.shape[0]
    width = input_tensor.shape[1]

    cur_row = 0
    cur_col = 0
    pixel_ind = 0
    while cur_row is not None:
        if input_tensor[cur_row, cur_col] == 1:
            cc_ind = find_neighbors_cc_ind(pixel_to_cc_ind, cc_to_final_cc, width, pixel_ind)
            if cc_ind is None:
                # New connected component
                cc_ind = next_cc_ind
                next_cc_ind += 1
            pixel_to_cc_ind[pixel_ind] = cc_ind

        cur_row, cur_col = advance_pixel(cur_row, cur_col, height, width)
        pixel_ind += 1

    return pixel_to_cc_ind, cc_to_final_cc, next_cc_ind


def find_cc_edges(pixel_to_cc_ind, height, width, wanted_cc_ind):
    cc_pixels = []
    for pixel_ind, cc_ind in pixel_to_cc_ind.items():
        if cc_ind == wanted_cc_ind:
            cc_pixels.append((pixel_ind // width, pixel_ind % width))
    upper_edge = height
    lower_edge = 0
    left_edge = width
    right_edge = 0
    for row, col in cc_pixels:
        if row < upper_edge:
            upper_edge = row
        if row > lower_edge:
            lower_edge = row
        if col < left_edge:
            left_edge = col
        if col > right_edge:
            right_edge = col

    return upper_edge, lower_edge, left_edge, right_edge


def find_largest_connected_component(input_tensor):
    """ Find the edges of the largest connected component (cc) in a 2D tensor, where each pixel is
    either valued 0 or 1. A (cc) is defined as a set of 1-valued pixels, where each pixel is a
    neighbor of some other pixel in the set, and none of the pixels has a 1-valued neighbor outside
    the set.
    We go over all the pixels, row-by-row, and inside each row column-by-column. For each pixel
    with value 1, we check if the previously visited neighbors (up-left, up, up-right, left) are
    already assigned with a cc. If not- we assign a new cc. If yes- we check if multiple neighbors
    have cc, and if are assigned to different cc's- they are actually the same one (through the
    current pixel that connects both of them). So we map one of the cc's to the other, and will
    deal with this later.
    After we finish this traverse, we map each cc to its "final" cc: we follow the cc mapping until
    we reach a cc which is not mapped to any other cc. Then we calculate the size of each cc, to
    find the largest."""
    # First traverse all the pixels
    pixel_to_cc_ind, cc_to_final_cc, cc_num = map_pixel_to_cc(input_tensor)

    # Next, we need to create the final mapping
    cc_to_final_cc = create_final_cc_mapping(cc_num, cc_to_final_cc)
    # Change the pixel -> cc mapping to fit the final cc's we found
    pixel_to_cc_ind = \
        {x: cc_to_final_cc[pixel_to_cc_ind[x]] for x in pixel_to_cc_ind.keys()}

    # Next, find the largest cc
    largest_cc_ind = argmax_cc(pixel_to_cc_ind)

    # Now, find the largest cc's edges
    height = input_tensor.shape[0]
    width = input_tensor.shape[1]
    largest_cc_edges = find_cc_edges(pixel_to_cc_ind, height, width, largest_cc_ind)

    return largest_cc_edges


def resize_activation_map(activation_map):
    upsample_op = nn.Upsample(size=wanted_image_size, mode='bicubic', align_corners=False)
    activation_map = activation_map.view(1, 1, activation_map.shape[0], activation_map.shape[1])
    return upsample_op(activation_map).view(wanted_image_size)


def predict_bbox(activation_map, segment_threshold_rate=0.5):
    orig_image_activation_vals = resize_activation_map(activation_map)

    cam_max_val = torch.max(orig_image_activation_vals)

    segmentation_map = torch.zeros(wanted_image_size)
    segmentation_map[orig_image_activation_vals >= segment_threshold_rate * cam_max_val] = 1

    upper_edge, lower_edge, left_edge, right_edge = \
        find_largest_connected_component(segmentation_map)

    bbox = [left_edge, upper_edge, right_edge, lower_edge]

    return bbox


def plot_bboxes(image_id, bbox_list):
    image_obj = Image.open(get_image_path(image_id))
    image_obj = image_obj.resize(wanted_image_size)

    draw = ImageDraw.Draw(image_obj)
    for bbox in bbox_list:
        draw.rectangle(bbox)

    plt.imshow(image_obj)
    plt.show()


def plot_heatmap(image_id, activation_map, explicitly_upsample):
    image_obj = Image.open(get_image_path(image_id))
    image_obj = image_obj.resize(wanted_image_size)
    if explicitly_upsample:
        heatmap = to_pil_image(resize_activation_map(activation_map), mode='F')
    else:
        heatmap = to_pil_image(activation_map, mode='F')
    result = overlay_mask(image_obj, heatmap)

    plt.imshow(result)
    plt.show()


def test_epoch_bbox(test_set, model):
    test_image_num = len(test_set)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in trange(test_image_num):
        image_id = list(test_set.keys())[i]

        gt_box_data = test_set[image_id]
        gt_class_labels = [x[1] for x in gt_box_data]
        unique_gt_class_labels = list(set(gt_class_labels))

        image_tensor = transform_to_torch_format(image_id)
        if image_tensor is None:
            continue

        output = model(image_tensor)
        predicted_classes = predict_classes(output)
        # CHANGE
        from torchvision.models import resnet18
        from torchcam.cams import CAM
        tmp_model = resnet18(pretrained=True).eval()
        tmp_cam_extractor = CAM(tmp_model)
        tmp_output = tmp_model(image_tensor)
        tmp_activation_map = tmp_cam_extractor(tmp_output.squeeze(0).argmax().item(), tmp_output)

        bbox1 = predict_bbox(tmp_activation_map, 0.5)
        bbox2 = predict_bbox(tmp_activation_map, 0.6)
        bbox3 = predict_bbox(tmp_activation_map, 0.7)
        bbox4 = predict_bbox(tmp_activation_map, 0.8)

        plot_bboxes(image_id, [bbox1, bbox2, bbox3, bbox4])
        plot_heatmap(image_id, tmp_activation_map, True)
        plot_heatmap(image_id, tmp_activation_map, False)

        for class_ind in predicted_classes:
            activation_map = model.extract_cam(class_ind, output)
            bbox1 = predict_bbox(activation_map, 0.5)
            bbox2 = predict_bbox(activation_map, 0.6)
            bbox3 = predict_bbox(activation_map, 0.7)
            bbox4 = predict_bbox(activation_map, 0.8)

            plot_bboxes(image_id, [bbox1, bbox2, bbox3, bbox4])
            plot_heatmap(image_id, activation_map, True)
            plot_heatmap(image_id, activation_map, False)
        # END CHANGE
        predicted_boxes_data = predict_bboxes(model, output, predicted_classes)
        tp, fp, fn = evaluate_bbox_prediction(predicted_boxes_data, gt_box_data)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    print('tp: ' + str(total_tp))


def test_epoch_classification(test_set, model):
    function_name = 'test_epoch_classification'
    indent = 2
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    dataloader = data.DataLoader(test_set, batch_size=1, shuffle=False)
    checkpoint_len = 1000
    checkpoint_time = time.time()
    for i_batch, sampled_batch in enumerate(dataloader):
        if i_batch % checkpoint_len == 0:
            log_print(function_name, indent + 1, 'Starting batch ' + str(i_batch) +
                      ' out of ' + str(len(dataloader)) +
                      ', time from previous checkpoint ' + str(time.time() - checkpoint_time))
            checkpoint_time = time.time()

        image_tensor = sampled_batch['image'].to(model.device)
        output = model(image_tensor)
        predicted_classes = predict_classes(output)

        gt_vector = sampled_batch['label']
        class_num = model.class_num
        gt_classes = [x for x in range(class_num) if gt_vector[0, x] == 1]

        tp, fp, tn, fn = evaluate_classification(predicted_classes, gt_classes, model.class_num)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    log_print(function_name, indent, 'Precision: ' + str(precision))
    log_print(function_name, indent, 'Recall: ' + str(recall))
    log_print(function_name, indent, 'F1: ' + str(f1))


def predict_classes(output):
    threshold = 0.3
    class_num = output.shape[1]
    prob_output = torch.sigmoid(output)
    return [x for x in range(class_num) if prob_output[0, x] > threshold]


def predict_bboxes(model, output, predicted_classes):
    for class_ind in predicted_classes:
        activation_map = model.extract_cam(class_ind, output)
        print('a')


def evaluate_classification(predicted_classes, gt_classes, class_num):
    predicted_num = len(predicted_classes)
    tp = len(list(set(predicted_classes).intersection(gt_classes)))
    fp = predicted_num - tp

    non_predicted_num = class_num - predicted_num
    fn = len(list(set(gt_classes).difference(predicted_classes)))
    tn = non_predicted_num - fn

    return tp, fp, tn, fn


def train_cam(timestamp, training_set, test_set, model, epoch_num):
    function_name = 'train_cam'
    indent = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    log_print(function_name, indent, 'Starting training...')
    for i in range(epoch_num):
        log_print(function_name, indent+1, 'Starting training Epoch ' + str(i+1))
        train_epoch(training_set, model, optimizer)
    log_print(function_name, indent, 'Finished training')

    model_path = os.path.join(timestamp, 'model.mdl')
    torch.save(model.state_dict(), model_path)
    # model.load_state_dict(torch.load('model.mdl'))
    log_print(function_name, indent, 'Testing...')
    test_epoch_classification(test_set, model)
    # test_epoch_bbox(test_set, model)
    log_print(function_name, indent, 'Finished testing')

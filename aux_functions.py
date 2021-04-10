import os
import torch

write_to_log = False
log_fp = None


def set_write_to_log(output_dir):
    global write_to_log
    global log_fp
    write_to_log = True
    log_path = os.path.join(output_dir, 'log.txt')
    log_fp = open(log_path, 'w')


def log_print(function_name, indent, my_str):
    if function_name == '':
        prefix = ''
    else:
        prefix = '[' + function_name + '] '
    full_str = '\t' * indent + prefix + my_str
    if write_to_log:
        log_fp.write(full_str + '\n')
        log_fp.flush()
    else:
        print(full_str)


def generate_dataset(filepath, generation_func, *args):
    if os.path.exists(filepath):
        dataset = torch.load(filepath)
    else:
        dataset = generation_func(*args)
        torch.save(dataset, filepath)
    return dataset


def calc_ious(box1, box2):
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    area_sum = box1_area.reshape((-1, 1)) + box2_area.reshape((1, -1))

    intersection_x_min = torch.max(box1[:, 0].reshape((-1, 1)), box2[:, 0].reshape((1, -1)))
    intersection_x_max = torch.min(box1[:, 2].reshape((-1, 1)), box2[:, 2].reshape((1, -1)))
    intersection_y_min = torch.max(box1[:, 1].reshape((-1, 1)), box2[:, 1].reshape((1, -1)))
    intersection_y_max = torch.min(box1[:, 3].reshape((-1, 1)), box2[:, 3].reshape((1, -1)))

    intersection_width = torch.max(intersection_x_max - intersection_x_min, torch.tensor([0.]))
    intersection_height = torch.max(intersection_y_max - intersection_y_min, torch.tensor([0.]))
    intersection_area = intersection_width * intersection_height
    union_area = area_sum - intersection_area
    ious = intersection_area / union_area
    return ious

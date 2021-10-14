import os
import torch
import time
from datetime import datetime

write_to_log = False
log_fp = None

models_dir = 'models'
visual_dir = 'visual'
text_dir = 'text'
default_model_name = 'model'


def set_write_to_log(output_dir):
    global write_to_log
    global log_fp
    write_to_log = True
    log_path = os.path.join(output_dir, 'log.txt')
    log_fp = open(log_path, 'w')


def log_print(class_name, indent, my_str):
    if class_name == '':
        prefix = ''
    else:
        prefix = '[' + class_name + '] '
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


def for_loop_with_reports(iterable, iterable_size, checkpoint_len, inner_impl, progress_report_func):
    checkpoint_time = time.time()

    for index, item in enumerate(iterable):
        should_print = False

        if index % checkpoint_len == 0:
            time_from_prev_checkpoint = time.time() - checkpoint_time
            progress_report_func(index, iterable_size, time_from_prev_checkpoint)
            checkpoint_time = time.time()
            should_print = True

        inner_impl(index, item, should_print)


def get_timestamp_str():
    return str(datetime.now()).replace(' ', '_').replace(':', '-')

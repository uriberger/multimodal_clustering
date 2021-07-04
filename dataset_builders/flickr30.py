import os
from xml.dom import minidom
from utils.general_utils import generate_dataset

flickr30_dir_path = os.path.join('../..', 'cached_dataset_files', 'flickr30')

tokens_dir_name = 'tokens'
tokens_file_name = 'results_20130124.token'
tokens_file_path = os.path.join(flickr30_dir_path, tokens_dir_name, tokens_file_name)

bbox_dir_name = os.path.join('annotations', 'Annotations')
bbox_dir_path = os.path.join(flickr30_dir_path, bbox_dir_name)

sentences_dir_name = os.path.join('annotations', 'Sentences')
sentences_dir_path = os.path.join(flickr30_dir_path, sentences_dir_name)

images_dir_name = 'images'
images_dir_path = os.path.join(flickr30_dir_path, images_dir_name)


def generate_captions():

    fp = open(tokens_file_path)
    img_to_caption_list = {}
    for line in fp:
        split_line = line.strip().split('#')
        img_file_name = split_line[0]
        caption = split_line[1].split(' ')[1:]  # The first token is caption number

        caption = [x for x in caption if len(x) > 0]

        if img_file_name not in img_to_caption_list:
            img_to_caption_list[img_file_name] = []
        img_to_caption_list[img_file_name].append(caption)

    return img_to_caption_list


def print_caption_statistics(img_to_caption_list):
    img_number = len(img_to_caption_list)
    print('Extracted captions for ' + str(img_number) + ' images')
    caption_number = sum([len(x) for x in img_to_caption_list.values()])
    print('Extracted ' + str(caption_number) + ' captions')
    token_number = sum([sum([len(y) for y in x]) for x in img_to_caption_list.values()])
    unique_tokens = {}
    for _, caption_list in img_to_caption_list.items():
        for caption in caption_list:
            for token in caption:
                unique_tokens[token] = True
    unique_token_number = len(unique_tokens)
    print('Overall ' + str(token_number) + ' tokens, ' + str(unique_token_number) + ' unique tokens')


coord_strs = ['xmin', 'ymin', 'xmax', 'ymax']
coord_str_to_ind = {coord_strs[x]: x for x in range(len(coord_strs))}
img_bboxes_dataset_filename = 'flickr30_img_bboxes_dataset'
boxes_chains_filename = 'flickr30_boxes_chains'
chains_and_classes_filename = 'flickr30_chains_and_classes'


def generate_bboxes_dataset_flickr30():
    """ This function returns two data structures:
        - A map of image id -> list of bbox, category id pairs (the actual dataset).
        - A category id -> name mapping, for mapping the labels to category names. """
    return generate_dataset(img_bboxes_dataset_filename, generate_bboxes_dataset_flickr30_internal)


def generate_bboxes_dataset_flickr30_internal():
    boxes_chains, chain_list = extract_boxes_and_chains()
    chain_to_class_ind, class_ind_to_str = get_chain_to_class_mapping(chain_list)
    img_bboxes_dataset = {y: [(x[0], chain_to_class_ind[x[1]]) for x in boxes_chains[y]] for y in boxes_chains.keys()}

    return img_bboxes_dataset, class_ind_to_str


def extract_boxes_and_chains():
    return generate_dataset(boxes_chains_filename, extract_boxes_and_chains_internal)


def extract_boxes_and_chains_internal():
    extracted_chains = {}
    boxes_chains = {}
    box_count = 0
    print('Extracting bounding boxes and coreference chains...')
    for _, _, files in os.walk(bbox_dir_path):
        file_ind = 0
        for filename in files:
            if file_ind % 10000 == 0:
                print('\tFile ' + str(file_ind) + ' out of ' + str(len(files)))

            # Extract image file name from current file name
            image_id = filename.split('.')[0]

            # Extract bounding boxes from file
            bounding_boxes = []

            xml_filepath = os.path.join(bbox_dir_path, filename)
            xml_doc = minidom.parse(xml_filepath)
            for child_node in xml_doc.childNodes[0].childNodes:
                # The bounding boxes are located inside a node named "object"
                if child_node.nodeName == u'object':
                    # Go over all of the children of this node: if we find bndbox, this object is a bounding box
                    box_chain = None
                    for inner_child_node in child_node.childNodes:
                        if inner_child_node.nodeName == u'name':
                            box_chain = int(inner_child_node.childNodes[0].data)
                        if inner_child_node.nodeName == u'bndbox':
                            # This is a bounding box node
                            box_count += 1
                            bounding_box = [None, None, None, None]
                            for val_node in inner_child_node.childNodes:
                                node_name = val_node.nodeName
                                if node_name in coord_strs:
                                    coord_ind = coord_str_to_ind[node_name]
                                    bounding_box[coord_ind] = int(val_node.childNodes[0].data)

                            # Check that all coordinates were found
                            none_inds = [x for x in range(len(bounding_box)) if x is None]
                            bounding_box_ind = len(bounding_boxes)
                            if len(none_inds) > 0:
                                for none_ind in none_inds:
                                    print('Didn\'t find coordinate ' + coord_strs[none_ind] + ' for bounding box ' +
                                          str(bounding_box_ind) + ' in image ' + filename)
                                assert False
                            if box_chain is None:
                                print('Didn\'t find chain for bounding box ' +
                                      str(bounding_box_ind) + ' in image ' + filename)
                                assert False
                            bounding_boxes.append((bounding_box, box_chain))

                            # Document chain
                            if box_chain not in extracted_chains:
                                extracted_chains[box_chain] = True
            boxes_chains[image_id] = bounding_boxes

            file_ind += 1

    print('Extracted bounding boxes and coreference chains')
    print('Found ' + str(len(boxes_chains)) + ' images, ' + str(box_count) + ' bounding boxes, ' + str(len(extracted_chains)) + ' coreference chains')
    chain_list = list(extracted_chains.keys())

    return boxes_chains, chain_list


def get_chain_to_class_mapping(chain_list):
    return generate_dataset(chains_and_classes_filename, get_chain_to_class_mapping_internal, chain_list)


def get_chain_to_class_mapping_internal(chain_list):
    chain_to_class_ind = {}
    class_str_to_ind = {}
    found_chains = {x: False for x in chain_list}
    print('Extracting chain to class mapping...')
    file_names = os.listdir(sentences_dir_path)
    for file_ind in range(len(file_names)):
        if file_ind % 10000 == 0:
            print('\tFile ' + str(file_ind) + ' out of ' + str(len(file_names)))

        # Extract annotated sentences from file
        filename = file_names[file_ind]
        filepath = os.path.join(sentences_dir_path, filename)
        fp = open(filepath, 'r')
        for line in fp:
            split_by_annotations = line.split('[/EN#')[1:]
            for line_part in split_by_annotations:
                annotation = line_part.split()[0].split('/')
                chain_ind = int(annotation[0])
                class_str = annotation[1]

                if class_str in class_str_to_ind:
                    class_ind = class_str_to_ind[class_str]
                else:
                    class_ind = len(class_str_to_ind)
                    class_str_to_ind[class_str] = class_ind

                chain_to_class_ind[chain_ind] = class_ind
                if chain_ind in found_chains and not found_chains[chain_ind]:
                    found_chains[chain_ind] = True

    # Add the 'unknown' class for chains we couldn't find
    unknown_class_ind = len(class_str_to_ind)
    class_str_to_ind['unknown'] = unknown_class_ind
    chains_not_found = [x for x in chain_list if not found_chains[x]]
    for chain_ind in chains_not_found:
        chain_to_class_ind[chain_ind] = unknown_class_ind

    print('Extracted chain to class mapping')
    class_ind_to_str = {class_str_to_ind[k]: k for k in class_str_to_ind.keys()}

    return chain_to_class_ind, class_ind_to_str


def get_image_path(image_id):
    image_filename = str(image_id) + '.jpg'
    image_path = os.path.join(images_dir_path, image_filename)

    return image_path

# img_to_caption_list = generate_captions()
# print_caption_statistics(img_to_caption_list)

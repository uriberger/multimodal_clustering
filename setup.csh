$1 -m spacy download en_core_web_sm
$1 main.py --utility filter_unwanted_coco_images --write_to_log --datasets_dir $2

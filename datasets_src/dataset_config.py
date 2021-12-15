###############################################
### Unsupervised Multimodal Word Clustering ###
### as a First Step of Language Acquisition ###
###############################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from loggable_object import LoggableObject


class DatasetConfig(LoggableObject):
    """ Contains configuration settings for datasets.

        simplified_captions: If true, captions are replaced by the concatenation of names of object class in the image,
        separated by spaces

        include_gt_classes/bboxes: A flag indicating whether we should generate the ground-truth data for classes or
        bounding boxes

        use_transformations: Use random transformations (rotations, cropping, etc.) on images in the dataset

        lemmatize: Lemmeatize the tokens in the captions

        slice_str: Which part of the dataset is it (e.g., train)
    """

    def __init__(self,
                 indent,
                 simplified_captions=False,
                 include_gt_classes=False,
                 include_gt_bboxes=False,
                 use_transformations=False,
                 lemmatize=False,
                 slice_str='train'
                 ):
        super(DatasetConfig, self).__init__(indent)

        # Check constraints
        if simplified_captions and not include_gt_classes:
            self.log_print('Can\'t get simplified captions without including gt_classes info!')
            assert False

        self.simplified_captions = simplified_captions
        self.include_gt_classes = include_gt_classes
        self.include_gt_bboxes = include_gt_bboxes
        self.use_transformations = use_transformations
        self.lemmatize = lemmatize
        self.slice_str = slice_str

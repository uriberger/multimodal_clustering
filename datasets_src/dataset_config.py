from loggable_object import LoggableObject


class DatasetConfig(LoggableObject):
    def __init__(self,
                 indent,
                 simplified_captions=False,
                 include_gt_classes=False,
                 include_gt_bboxes=False,
                 use_transformations=False,
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
        self.slice_str = slice_str

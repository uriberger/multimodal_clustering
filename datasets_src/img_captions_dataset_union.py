import torch.utils.data as data
import torch


class ImageCaptionDatasetUnion(data.Dataset):
    """An implementation of union of several datasets of the ImageCaptionDataset class.
    Currently not implemented with gt classes or bboxes.
    """

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.dataset_sizes = [len(dataset) for dataset in self.dataset_list]
        self.sample_indices = []
        for i in range(len(self.dataset_sizes)):
            self.sample_indices += [(i, x) for x in range(self.dataset_sizes[i])]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dataset_index, sample_index = self.sample_indices[idx]
        return self.dataset_list[dataset_index].__getitem__(sample_index)

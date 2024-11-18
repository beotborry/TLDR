from torch.utils.data import Dataset
from datasets.base_spuco_compatible_dataset import BaseSpuCoCompatibleDataset
from datasets.imagenet_dataset import ImagenetDataset
from typing import Union

class ProjDatasetWrapper:
    def __init__(
    self,
    dataset: Union[BaseSpuCoCompatibleDataset, ImagenetDataset],
    ):
        self.dataset = dataset
        self.modality = dataset.modality

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        """
        source_tuple = self.dataset.__getitem__(index)
        return source_tuple[0], source_tuple[1] # img and label if modality is not "both", else img and text
    
    def __len__(self):
        """
        Returns the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.dataset)
import torch
import numpy as np
import bisect
from torch.utils.data import Dataset
from datasets.base_spuco_compatible_dataset import BaseSpuCoCompatibleDataset
from datasets.imagenet_dataset import ImagenetDataset
from datasets.text_dataset import TextDataset
from typing import Union
from typing import Dict, List, Tuple
from tqdm import tqdm

class RectDatasetWrapper(Dataset):
    def __init__(
    self,
    dataset: Union[BaseSpuCoCompatibleDataset, TextDataset],
    ):
        self.dataset = dataset
        self.labels_ = self.dataset.labels
        self.spurious_ = self.dataset.spurious
        self.group_partition_ = self.dataset.group_partition
        self.group_weights_ = self.dataset.group_weights
        self.num_classes_ = self.dataset.num_classes

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        """
        source_tuple = self.dataset.__getitem__(index)

        return source_tuple[0], source_tuple[1]
    
    def __len__(self):
        """
        Returns the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.dataset)
    

    @property
    def group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        return self.group_partition_

    @property
    def group_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Dictionary containing the fractional weights of each group
        """
        return self.group_weights_

    @property
    def spurious(self) -> List[int]:
        """
        List containing spurious labels for each example
        """
        return self.spurious_

    @property
    def labels(self) -> List[int]:
        """
        List containing class labels for each example
        """
        return self.labels_

    @property
    def num_classes(self) -> int:
        """
        Number of classes
        """
        return self.num_classes_
    
class RectConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, group_balance=False):
        super(RectConcatDataset, self).__init__(datasets)

        self.text_dataset = datasets[0]
        self.image_dataset = datasets[1]

        self.labels_ = self.text_dataset.labels + self.image_dataset.labels
        self.spurious_ = self.text_dataset.spurious + self.image_dataset.spurious
        self.num_classes_ = self.text_dataset.num_classes

        self.group_balance = group_balance

        self._group_partition = {}

        for i, group_label in tqdm(
            enumerate(zip(self._labels, self._spurious)),
            desc="Partitioning data indices into groups",
            disable=True,
            total=len(self.data)
        ):
            if group_label not in self._group_partition:
                self._group_partition[group_label] = []
            self._group_partition[group_label].append(i)
        
        # Set group weights based on group sizes
        self._group_weights = {}
        for group_label in self._group_partition.keys():
            self._group_weights[group_label] = len(self._group_partition[group_label]) / len(self.data)

        if self.group_balance:
            min_size = min([len(self._group_partition[group_label]) for group_label in self._group_partition.keys()])
            self.indices = []
            for group_label in self._group_partition.keys():
                np.random.shuffle(self._group_partition[group_label])
                self.indices += self._group_partition[group_label][:min_size]
    
    def __len__(self):
        if self.group_balance:
            return len(self.indices)
        else:
            return len(self.text_dataset) + len(self.image_dataset)
        
    def __getitem__(self, idx):
        if self.group_balance:
            idx = self.indices[idx]
            
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        return self.group_partition

    @property
    def group_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Dictionary containing the fractional weights of each group
        """
        return self.group_weights

    @property
    def spurious(self) -> List[int]:
        """
        List containing spurious labels for each example
        """
        return self.spurious_

    @property
    def labels(self) -> List[int]:
        """
        List containing class labels for each example
        """
        return self.labels_

    @property
    def num_classes(self) -> int:
        """
        Number of classes
        """
        return self.num_classes_
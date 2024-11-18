import torch
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
from datasets.wilds_dataset_wrapper import WILDSDatasetWrapper
from datasets.spuco_animals import SpuCoAnimals
from datasets.no_minority_group_wrapper import NoMinorityGroupWrapper

class Subset(torch.utils.data.Dataset):
    """
    Subsets a dataset while preserving original indexing.

    NOTE: torch.utils.dataset.Subset loses original indexing.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self._indices = indices
      
        if isinstance(dataset, WILDSDatasetWrapper) or isinstance(dataset, SpuCoAnimals):
            try: # wilds
                self._labels = [self.dataset._labels[i] for i in self._indices]
                self._spurious = [self.dataset._spurious[i] for i in self._indices]
            except: # spucoanimals
                self._labels = [self.dataset.data.labels[i] for i in self._indices]
                self._spurious = [self.dataset.data.spurious[i] for i in self._indices]
        elif isinstance(dataset, NoMinorityGroupWrapper):
            self._labels = [self.dataset.dataset.labels[self.dataset.filtered_indices[i]] for i in self._indices]
            self._spurious = [self.dataset.dataset.spurious[self.dataset.filtered_indices[i]] for i in self._indices]

        self.num_classes_ = len(set(self._labels))

        # Create group partition using labels and spurious labels
        self._group_partition = {}
        for i, group_label in tqdm(
            enumerate(zip(self._labels, self._spurious)),
            desc="Partitioning data indices into groups",
            disable=False,
            total=len(self._indices)
        ):
            if group_label not in self._group_partition:
                self._group_partition[group_label] = []
            self._group_partition[group_label].append(i)
        
        # Set group weights based on group sizes
        self._group_weights = {}
        for group_label in self._group_partition.keys():
            self._group_weights[group_label] = len(self._group_partition[group_label]) / len(self._indices)

    def __getitem__(self, idx):
        return self.dataset[self._indices[idx]]

    def __len__(self):
        return len(self._indices)

    @property
    def group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        return self._group_partition
    
    @property
    def group_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Dictionary containing the fractional weights of each group
        """
        return self._group_weights
    
    @property
    def spurious(self) -> List[int]:
        """
        List containing spurious labels for each example
        """
        return self._spurious

    @property
    def labels(self) -> List[int]:
        """
        List containing class labels for each example
        """
        return self._labels
    
    @property
    def index(self) -> List[int]:
        """
        List containing indices for each example
        """
        return self._indices
    
    @property
    def num_classes(self) -> int:
        """
        Number of classes
        """
        return self.num_classes_
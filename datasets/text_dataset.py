import random
from typing import Dict, List, Optional, Tuple
import clip
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class TextDataset(Dataset):
    """
    Text Dataset.
    The data structure is assumed to be:
    - data: List[Dict]
        A list of dictionaries, each dictionary contains:
        - text: str
        - label: int
    """

    def __init__(
        self,
        data: List[Dict],
        max_data_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.data = data
        self.max_data_size = max_data_size
        self.modality = "text"

        if self.max_data_size is not None and len(self.data) > self.max_data_size:
            random.seed(1234)
            indices = random.sample(range(len(self.data)), self.max_data_size)
            self.data = [self.data[i] for i in indices]

        self._labels = np.array([data[idx].get("label", None) for idx in range(len(self.data))])
        self._spurious = np.array([data[idx].get("spurious", None) for idx in range(len(self.data))])

        # Create group partition using labels and spurious labels
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

        self.sample_weight = []
        for i in range(len(self.data)):
            self.sample_weight.append(1 / len(self._group_partition[(self._labels[i], self._spurious[i])]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int, Dict]:
        label = self.data[idx].get("label", None)

        # return text, label
        return self.data[idx], label
    
    @property
    def labels(self):
        """
        List containing class labels for each example
        """
        return self._labels

    @property
    def spurious(self):
        """
        List containing spurious labels for each example
        """
        return self._spurious
    

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
    def num_classes(self) -> int:
        """
        Number of classes
        """
        return len(set(self.labels))
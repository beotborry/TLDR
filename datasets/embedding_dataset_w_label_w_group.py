from torch.utils.data import Dataset

class EmbeddingDatasetWLabelWGroup(Dataset):
    def __init__(self, embeddings, labels, group_partition):
        self.embeddings = embeddings
        self.labels = labels
        self.group_partition = group_partition
        self.num_classes = len(set(labels))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
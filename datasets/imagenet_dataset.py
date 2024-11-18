from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# from torchvision.models import ResNet50_Weights
import os
import numpy as np


class ImagenetDataset(Dataset):
    def __init__(self, split, n_sample=None, root_dir="/home/juhyeon/data/Imagenet"):
        self.root_dir = os.path.join(root_dir, split)
        self.n_sample = int(n_sample) if n_sample is not None else None
        # self.transform = ResNet50_Weights.DEFAULT.transforms()
        self.modality = "image"
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.image_list = []
        self.label_list = []
        self.class_list = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_list.sort()

        if self.n_sample is not None:
            # sample n_sample images from each class
            for i, class_name in enumerate(self.class_list):
                class_dir = os.path.join(self.root_dir, class_name)
                class_image_list = [os.path.join(class_dir, d) for d in os.listdir(class_dir)]
                class_image_list = np.array(class_image_list)
                # shuffle class_image_list
                np.random.shuffle(class_image_list)
                self.image_list += class_image_list[: self.n_sample].tolist()
                self.label_list += [i] * self.n_sample
        else:
            for i, class_name in enumerate(self.class_list):
                class_dir = os.path.join(self.root_dir, class_name)
                class_image_list = [os.path.join(class_dir, d) for d in os.listdir(class_dir)]
                self.image_list += class_image_list
                self.label_list += [i] * len(class_image_list)

        self.num_classes = len(self.class_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, -1, idx

    def get_num_classes(self):
        return self.num_classes

    def get_class_list(self):
        return self.class_list

    def get_class_name(self, idx):
        return self.class_list[idx]

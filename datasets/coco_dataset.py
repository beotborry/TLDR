from pycocotools.coco import COCO
from torch.utils import data
from PIL import Image
from torchvision import transforms
from os.path import expanduser

import os
import numpy as np
import clip

class CoCoDataset(data.Dataset):
    def __init__(
        self,
        annotations_file,
        img_folder,
        n_sample,
        clip_variants = "ViT-B/32",
    ):
        home = expanduser("~")
        # self.root = os.path.join(home, 'fiftyone/coco-2014')
        self.root = "/data1/fiftyone/coco-2014"
        _, self.transform = clip.load(clip_variants, device="cpu")
        self.img_folder = os.path.join(self.root, img_folder)
        self.coco = COCO(os.path.join(self.root, annotations_file))
        self.ids = list(self.coco.anns.keys())
        np.random.shuffle(self.ids)
        self.ids = self.ids[:n_sample]
        self.modality = "both"
        print("Total number of samples: {}".format(len(self.ids)))

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]["caption"]
        caption = clip.tokenize(caption).squeeze(0)
        img_id = self.coco.anns[ann_id]["image_id"]
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        
        image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
        image = self.transform(image)
        return image, caption
    
    def __len__(self):
        return len(self.ids)
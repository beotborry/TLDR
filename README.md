# TLDR
Official Implementation of TLDR: Text Based Last-layer Retraining for Debiasing Image Classifiers (WACV 2025) (arXiv: https://arxiv.org/abs/2311.18291)

This code is based on the SpuCo repository (https://github.com/BigML-CS-UCLA/SpuCo)

# Environment Setup
```
conda env create --file tldr.yaml
```

# Training
- The code will automatically download Waterbirds, CelebA, and SpucoAnimals dataset.
- The code assume that COCO-2014 dataset is located at `/data1/fiftyone/coco-2014`. You can modify this path by changing https://github.com/beotborry/TLDR/blob/63bdf5ca1e5b4e6fa192f5096a5819fff6f61a6f/datasets/coco_dataset.py#L21C9-L21C48.
  
```
bash script.sh
```

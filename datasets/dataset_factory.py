import torchvision.transforms as transforms
import clip
from wilds import get_dataset as get_dataset_wilds
from datasets.wilds_dataset_wrapper import WILDSDatasetWrapper
from datasets.spuco_animals import SpuCoAnimals
class DatasetFactory:
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        augment: bool,
        args,
    ):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.args = args
        self.augment = augment

    def get_transform(self):
        if self.dataset_name == "waterbirds":
            # From https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/wb_data.py
            target_resolution = (224, 224)
            scale = 256.0 / 224.0

            if not self.augment:
                self.transform_train = transforms.Compose(
                    [
                        transforms.Resize(
                            (
                                int(target_resolution[0] * scale),
                                int(target_resolution[1] * scale),
                            )
                        ),
                        transforms.CenterCrop(target_resolution),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

                self.transform_test = self.transform_train
            else:
                self.transform_train = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            (224, 224),
                            scale=(0.7, 1.0),
                            ratio=(0.75, 1.3333333333333333),
                            interpolation=2,
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

                self.transform_test = transforms.Compose(
                    [
                        transforms.Resize(
                            (
                                int(target_resolution[0] * scale),
                                int(target_resolution[1] * scale),
                            )
                        ),
                        transforms.CenterCrop(target_resolution),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

        elif self.dataset_name == "spuco_animal":
            self.transform = transforms.Compose(
                [
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        elif self.dataset_name == "celeba":
            # From https://github.com/anniesch/jtt/blob/master/data/celebA_dataset.py
            orig_w = 178
            orig_h = 218
            orig_min_dim = min(orig_w, orig_h)
            target_resolution = (224, 224)

            if not self.augment:
                self.transform_train = transforms.Compose(
                    [
                        transforms.CenterCrop(orig_min_dim),
                        transforms.Resize(target_resolution),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
                self.transform_test = self.transform_train
            else:
                self.transform_train = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            target_resolution,
                            scale=(0.7, 1.0),
                            ratio=(1.0, 1.3333333333333333),
                            interpolation=2,
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

                self.transform_test = transforms.Compose(
                    [
                        transforms.CenterCrop(orig_min_dim),
                        transforms.Resize(target_resolution),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

    def get_dataset(self, clip_transform=False, dfr_new_bias_test=False):
        if clip_transform:
            _, transform = clip.load(self.args.clip_variants, device="cpu")
            self.transform_train = transform
            self.transform_test = transform
        else:
            self.get_transform()
            
        if self.dataset_name == "waterbirds":
            dataset = get_dataset_wilds(
                dataset="waterbirds",
                root_dir=self.root_dir,
                download=True,
            )

            train_data = dataset.get_subset("train", transform=self.transform_train)
            val_data = dataset.get_subset("val", transform=self.transform_test)
            test_data = dataset.get_subset("test", transform=self.transform_test)

            trainset = WILDSDatasetWrapper(
                dataset=train_data,
                metadata_spurious_label="background",
                verbose=True,
            )
            valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)
            testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)

        elif self.dataset_name == "spuco_animal":
            trainset = SpuCoAnimals(
                root=self.root_dir,
                label_noise=0,
                split="train",
                transform=self.transform,
            )
            trainset.initialize()

            valset = SpuCoAnimals(
                root=self.root_dir,
                label_noise=0,
                split="val",
                transform=self.transform,
            )
            valset.initialize()

            testset = SpuCoAnimals(
                root=self.root_dir,
                label_noise=0,
                split="test",
                transform=self.transform,
            )
            testset.initialize()


            print([len(x) for x in trainset.group_partition.values()], [len(x) for x in valset.group_partition.values()], [len(x) for x in testset.group_partition.values()])

        elif self.dataset_name == "celeba":
            dataset = get_dataset_wilds(
                dataset="celebA",
                root_dir=self.root_dir,
                download=True,
            )

            train_data = dataset.get_subset("train", transform=self.transform_train)
            val_data = dataset.get_subset("val", transform=self.transform_test)
            test_data = dataset.get_subset("test", transform=self.transform_test)

            trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="male", verbose=True)
            valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="male", verbose=True)
            testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label=self.args.dfr_new_bias_name.lower(), verbose=True)

            print("train group partition")
            for key, item in trainset.group_partition.items():
                print(key, len(item))
                
            print("val group partition")
            for key, item in valset.group_partition.items():
                print(key, len(item))
                
            print("test group partition")
            for key, item in testset.group_partition.items():
                print(key, len(item))   
                
        _, clip_transform = clip.load("ViT-B/32", device="cpu")

        return trainset, valset, testset

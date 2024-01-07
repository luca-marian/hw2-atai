from PIL import Image
import os
import pandas as pd

from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

from utils import natural_sort_key


class LabeledData(Dataset):
    def __init__(self, annotation_file, transform=None, options=None) -> None:
        if isinstance(annotation_file, pd.DataFrame):
            self._annotations = annotation_file
            self._annotations.columns = ["sample", "label"]
        else:
            self._annotations = pd.read_csv(annotation_file)
            self._annotations.columns = ["sample", "label"]
            self._annotations["sample"] = (
                options.dataset_root + "/" + self._annotations["sample"]
            )

        self.labels = self._annotations["label"]
        self.images_path = self._annotations["sample"]
        self.transform = transform

    def __getitem__(self, idx):
        _path_to_image = os.path.join(self.images_path[idx])
        image = Image.open(_path_to_image).convert("RGB")
        label = self.labels[idx].item()

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images_path)


class UnlabeledData(Dataset):
    def __init__(self, image_dir, transform=None) -> None:
        self.image_dir = image_dir
        self.images_path = sorted(os.listdir(image_dir), key=natural_sort_key)
        self.labels = [None for x in range(len(self.images_path))]
        self.transform = transform

    def __getitem__(self, idx):
        _path_to_image = os.path.join(self.image_dir, self.images_path[idx])
        image = Image.open(_path_to_image).convert("RGB")

        try:
            label = self.labels[idx].item()

        except:
            label = 1000

        if self.transform:
            image = self.transform(image)

        return image, label, _path_to_image

    def __len__(self):
        return len(self.images_path)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None) -> None:
        self.images_path = dataframe["sample"]
        self.labels = dataframe["label"]

        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.images_path.loc[idx]).convert("RGB")
        label = self.labels[idx].item()

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images_path)


def get_split_dataset(data, seed=42, test_size=0.2, batch_size=64):
    """Taken form https://stackoverflow.com/a/68338670"""

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(data)),
        data.labels,
        stratify=data.labels,
        test_size=test_size,
        random_state=seed,
    )

    # generate subset based on indices
    train_split = Subset(data, train_indices)
    test_split = Subset(data, test_indices)

    # create batches
    train_batches = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    test_batches = DataLoader(test_split, batch_size=batch_size)

    return train_batches, test_batches, len(train_indices), len(test_indices)

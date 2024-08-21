import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import numpy as np
import torch
import cfg

from torchvision import datasets, transforms, models


idx_to_cat = {0: 'angry', 1: 'sad', 2: 'disgusted',
              3: 'neutral', 4: 'fearful', 5: 'happy', 6: 'surprised'}
cat_to_idx = {'angry': 0, 'sad': 1, 'disgusted': 2,
              'neutral': 3, 'fearful': 4, 'happy': 5, 'surprised': 6}


class Emotion(Dataset):
    base_folder = 'EmotionDetection/files'

    def __init__(self, root, train='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self._load_metadata()

    def _load_metadata(self):
        print(os.path.join(self.root, 'EmotionDetection', 'train.csv'))
        images = pd.read_csv(os.path.join(self.root, 'EmotionDetection', 'train.csv'), sep=',',
                             names=['ID', 'filepath'])

        train_images = pd.read_csv(os.path.join(
            self.root, 'EmotionDetection', 'train.csv'))

        test_images_no_labels = pd.read_csv(os.path.join(
            self.root, 'EmotionDetection', 'test.csv'))

        shuffle_indices = np.arange(len(train_images))
        seed = 42
        rng = np.random.RandomState(seed)
        rng.shuffle(shuffle_indices)
        test_frac = 0.05
        n_test = int(test_frac * len(shuffle_indices))
        train_data = train_images.loc[shuffle_indices[:]]
        val_data = train_images.loc[shuffle_indices[:n_test]]
        print(
            f"Loaded:\n- Train: {len(train_data)}\n- Val: {len(val_data)}\n- Test: {len(test_images_no_labels)}")

        if self.train == 'train':
            self.data = train_data
        elif self.train == 'val':
            self.data = val_data
        elif self.train == 'test':
            self.data = test_images_no_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filename)
        if self.train in ['train', 'val']:
            target = cat_to_idx[sample.label]
        else:
            target = 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

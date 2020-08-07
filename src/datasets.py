import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from src.audio import read_as_melspectrogram
from src.utils import get_params_hash
from src import config


def make_spectrogram_and_save(file_path: Path, save_dir: Path, audio_params):
    spec = read_as_melspectrogram(file_path, audio_params)
    save_class_dir = save_dir / file_path.parents[0].name
    save_class_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_class_dir / (file_path.name + '.npy')
    np.save(save_path, spec)


def prepare_train_data(dir_path, audio_params):
    dir_path = Path(dir_path)
    file_path_lst = []
    train_df = pd.read_csv(config.train_folds_path)
    for i, row in train_df.iterrows():
        file_path = config.train_audio_dir / row.ebird_code / row.filename
        file_path_lst.append(file_path)

    func = partial(make_spectrogram_and_save,
                   save_dir=dir_path, audio_params=audio_params)

    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(func, file_path_lst)


def check_prepared_train_data(audio_params):
    params_hash = get_params_hash(audio_params.dict())
    prepared_train_dir = config.prepared_train_dir / params_hash

    if not prepared_train_dir.exists():
        print(f"Start preparing dataset to '{prepared_train_dir}'")
        prepare_train_data(prepared_train_dir, audio_params)
        print(f"Dataset prepared.")
    else:
        print(f"'{prepared_train_dir}' already exists.")


def get_folds_data(audio_params):
    params_hash = get_params_hash(audio_params.dict())
    prepared_train_dir = config.prepared_train_dir / params_hash
    train_df = pd.read_csv(config.train_folds_path)
    train_dict = train_df.to_dict(orient='index')
    folds_data = []
    for _, sample in train_dict.items():
        class_dir = prepared_train_dir / sample['ebird_code']
        sample['spec_path'] = str(class_dir / (sample['filename'] + '.npy'))
        folds_data.append(sample)
    return folds_data


class BirdsongDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 target=True,
                 transform=None,
                 mixer=None,
                 random_class=False):
        self.folds = folds
        self.target = target
        self.transform = transform
        self.mixer = mixer
        self.random_class = random_class

        self.data = data

        if folds is not None:
            self.data = [s for s in self.data if s['fold'] in folds]

        class2indexes = defaultdict(list)
        for idx, sample in enumerate(self.data):
            class2indexes[sample['ebird_code']] += [idx]
        self.class2indexes = dict(class2indexes)

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        if self.random_class:
            cls = np.random.choice(config.classes)
            idx = np.random.choice(self.class2indexes[cls])

        sample = self.data[idx]
        image = np.load(sample['spec_path'])
        target = torch.zeros(len(config.classes))
        target[config.class2target[sample['ebird_code']]] = 1.
        return image, target

    def _set_random_seed(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        seed = int(time.time() * 1000.0) + idx
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    @torch.no_grad()
    def __getitem__(self, idx):
        self._set_random_seed(idx)
        if not self.target:
            image = self.get_sample(idx)
            if self.transform is not None:
                image = self.transform(image)
            return image
        else:
            image, target = self.get_sample(idx)
            if self.transform is not None:
                image = self.transform(image)
            if self.mixer is not None:
                image, target = self.mixer(self, image, target)
            return image, target


if __name__ == "__main__":
    check_prepared_train_data(audio_params=config.audio)

import random
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial

from src.audio import read_as_melspectrogram
from src.utils import get_params_hash
from src import config


def make_spectrogram_and_save(file_path: Path, save_dir: Path, audio_params):
    spec = read_as_melspectrogram(file_path, audio_params)
    if spec.shape[1] >= 320:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / (file_path.name + '.npy')
        np.save(save_path, spec)


def prepare_freesound_data(dir_path, audio_params):
    dir_path = Path(dir_path)
    file_path_lst = []
    train_df = pd.read_csv(config.freesound_train_curated_csv_path)
    for i, row in train_df.iterrows():
        if 'Chirp_and_tweet' not in row.labels:
            file_path = config.freesound_train_curated_dir / row.fname
            file_path_lst.append(file_path)

    func = partial(make_spectrogram_and_save,
                   save_dir=dir_path, audio_params=audio_params)

    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(func, file_path_lst)


def check_prepared_freesound_data(audio_params):
    params_hash = get_params_hash(audio_params.dict())
    prepared_train_dir = config.freesound_prepared_train_curated_dir / params_hash

    if not prepared_train_dir.exists():
        print(f"Start preparing freesound dataset to '{prepared_train_dir}'")
        prepare_freesound_data(prepared_train_dir, audio_params)
        print(f"Dataset prepared.")
    else:
        print(f"'{prepared_train_dir}' already exists.")


def get_freesound_folds_data(audio_params):
    params_hash = get_params_hash(audio_params.dict())
    prepared_train_dir = config.freesound_prepared_train_curated_dir / params_hash

    folds_data = []
    audio_paths = sorted(prepared_train_dir.glob("*.npy"))
    random.Random(42).shuffle(audio_paths)
    for i, spec_path in enumerate(audio_paths):
        sample = {
            'ebird_code': 'nocall',
            'spec_path': spec_path,
            'fold': i % config.n_folds
        }
        folds_data.append(sample)
    return folds_data


if __name__ == "__main__":
    check_prepared_freesound_data(audio_params=config.audio)

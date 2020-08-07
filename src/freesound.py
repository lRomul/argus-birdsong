import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial

from src.audio import read_as_melspectrogram
from src.utils import get_params_hash
from src import config


NOISE_SOUNDS = [
    'Buzz',
    'Car_passing_by',
    'Crackle',
    'Cricket',
    'Hiss',
    'Mechanical_fan',
    'Stream',
    'Traffic_noise_and_roadway_noise',
    'Walk_and_footsteps',
    'Waves_and_surf',
]


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
        noise = True
        for label in row.labels.split(','):
            if label not in NOISE_SOUNDS:
                noise = False
                break

        if noise:
            file_path = config.freesound_train_curated_dir / row.fname
            file_path_lst.append(file_path)

    func = partial(make_spectrogram_and_save,
                   save_dir=dir_path, audio_params=audio_params)

    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(func, file_path_lst)


def check_prepared_freesound_data(audio_params):
    params_hash = get_params_hash(audio_params.dict())
    prepared_train_dir = config.freesound_prepared_train_curated_dir / params_hash
    shutil.rmtree(prepared_train_dir)

    print(f"Start preparing freesound dataset to '{prepared_train_dir}'")
    prepare_freesound_data(prepared_train_dir, audio_params)
    print(f"Dataset prepared.")


def get_freesound_folds_data(audio_params):
    params_hash = get_params_hash(audio_params.dict())
    prepared_train_dir = config.freesound_prepared_train_curated_dir / params_hash

    folds_data = []
    audio_paths = sorted(prepared_train_dir.glob("*.npy"))
    for i, spec_path in enumerate(audio_paths):
        sample = {
            'ebird_code': 'nocall',
            'spec_path': spec_path,
            'fold': config.n_folds
        }
        folds_data.append(sample)
    return folds_data


if __name__ == "__main__":
    check_prepared_freesound_data(audio_params=config.audio)

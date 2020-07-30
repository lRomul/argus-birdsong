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


if __name__ == "__main__":
    check_prepared_train_data(audio_params=config.audio)

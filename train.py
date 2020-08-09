import json
import argparse

from torch.utils.data import DataLoader

from argus import load_model
from argus.callbacks import (
    MonitorCheckpoint,
    EarlyStopping,
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR
)

from src.datasets import (
    BirdsongDataset,
    check_prepared_train_data,
    get_folds_data
)
from src.mixers import get_mixer
from src.transforms import get_transforms
from src.argus_models import BirdsongModel
from src.freesound import (
    get_freesound_folds_data,
    check_prepared_freesound_data
)
from src.utils import initialize_amp, get_best_model_path
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='', type=str)
args = parser.parse_args()

BATCH_SIZE = 64
EPOCHS = 50
CROP_SIZE = 320
MIXER_PROB = 0.0
NUM_WORKERS = 8
USE_AMP = True
ITER_SIZE = 1
FREESOUND = True
DEVICES = ['cuda:0', 'cuda:1']
SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('ResNeSt', {
        'model_name': 'resnest50_fast_1s1x64d',
        'pretrained': True,
        'num_classes': config.n_classes
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('AdamW', {'lr': 0.001}),
    'device': DEVICES[0],
    'iter_size': ITER_SIZE
}


def train_fold(save_dir, train_folds, val_folds, folds_data):
    train_transfrom = get_transforms(train=True,
                                     size=CROP_SIZE,
                                     wrap_pad_prob=0.0,
                                     resize_scale=(0.8, 1.0),
                                     resize_ratio=(1.7, 2.3),
                                     resize_prob=0.0,
                                     spec_num_mask=2,
                                     spec_freq_masking=0.15,
                                     spec_time_masking=0.20,
                                     spec_prob=0.0)
    val_transform = get_transforms(train=False, size=CROP_SIZE)

    if MIXER_PROB:
        mixer = get_mixer(mixer_prob=MIXER_PROB,
                          sigmoid_range=(3, 12),
                          alpha_dist='uniform',
                          random_prob=(0.6, 0.4))
    else:
        mixer = None

    train_dataset = BirdsongDataset(folds_data, folds=train_folds,
                                    transform=train_transfrom, mixer=mixer)
    val_dataset = BirdsongDataset(folds_data, folds=val_folds,
                                  transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2 // ITER_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = BirdsongModel(PARAMS)
    if 'pretrained' in model.params['nn_module'][1]:
        model.params['nn_module'][1]['pretrained'] = False

    if USE_AMP:
        initialize_amp(model)

    model.set_device(DEVICES)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_loss', max_saves=1),
        CosineAnnealingLR(T_max=5 * len(train_dataset), eta_min=0, step_on_iteration=True),
        EarlyStopping(monitor='val_loss', patience=12),
        LoggingToFile(save_dir / 'log.txt'),
        LoggingToCSV(save_dir / 'log.csv')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=EPOCHS,
              callbacks=callbacks,
              metrics=['f1_score'])

    del model

    model_path = get_best_model_path(save_dir)
    model = load_model(model_path)
    val_dataset = BirdsongDataset(folds_data, folds=val_folds + [config.n_folds],
                                  transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2 // ITER_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    model.set_device(DEVICES[0])
    model.validate(val_loader, metrics=['f1_score'],
                   callbacks=[LoggingToFile(save_dir / 'log.txt'),
                              LoggingToCSV(save_dir / 'log.csv')])


if __name__ == "__main__":
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    print("Model params", PARAMS)
    with open(SAVE_DIR / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    check_prepared_train_data(config.audio)
    folds_data = get_folds_data(config.audio)

    if FREESOUND:
        check_prepared_freesound_data(config.audio)
        folds_data += get_freesound_folds_data(config.audio)

    if args.folds:
        folds = [int(fold) for fold in args.folds.split(',')]
    else:
        folds = config.folds

    for fold in folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data)

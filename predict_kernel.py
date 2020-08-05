import warnings
import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.transforms import get_transforms
from src.audio import read_audio, audio_to_melspectrogram
from src.utils import get_best_model_path

from src import config


warnings.filterwarnings('ignore',
                        'PySoundFile failed. Trying audioread instead.',
                        UserWarning)

EXPERIMENT = "stride_002"
CROP_SIZE = 320
BATCH_SIZE = 16
THRESHOLD = 0.2
DEVICE = 'cuda'


def prepare_audio_id_to_spec_data(test_df, audio_params):
    audio_id2spec = dict()
    for audio_id in sorted(test_df.audio_id.unique()):
        file_path = config.test_audio_dir / (audio_id + '.mp3')
        audio, sr = read_audio(file_path, audio_params.sampling_rate)
        spec = audio_to_melspectrogram(audio, audio_params)
        audio_id2spec[audio_id] = spec
    return audio_id2spec


def fold_pred(predictor, audio_id2spec):
    audio_id2pred = dict()
    for audio_id, spec in audio_id2spec.items():
        pred = predictor.predict(spec)
        audio_id2pred[audio_id] = pred
    return audio_id2pred


def experiment_pred(experiment_dir, audio_id2spec):
    print(f"Start predict: {experiment_dir}")
    transforms = get_transforms(False, CROP_SIZE)

    pred_lst = []
    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = experiment_dir / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = Predictor(model_path, transforms, BATCH_SIZE,
                              CROP_SIZE, CROP_SIZE, DEVICE)

        transforms = get_transforms(False, CROP_SIZE)

        pred = fold_pred(predictor, audio_id2spec)
        pred_lst.append(pred)

    audio_id2pred = dict()
    for audio_id in audio_id2spec:
        pred = [p[audio_id] for p in pred_lst]
        audio_id2pred[audio_id] = np.mean(pred, axis=0)

    return audio_id2pred


def pred2classes(pred, threshold=0.5):
    targets = np.argwhere(pred >= threshold)
    targets = targets.reshape(-1).tolist()
    classes = [config.target2class[t] for t in targets]
    return classes


def make_submission(test_df, audio_id2pred):
    row_id_lst = []
    birds_lst = []
    for audio_id, group in test_df.groupby('audio_id'):
        group = group.sort_values('seconds')

        site = group.site.unique().tolist()
        assert len(site) == 1
        site = site[0]

        audio_pred = audio_id2pred[audio_id]

        if site != 'site_3':
            for i, row in group.reset_index().iterrows():
                classes = pred2classes(audio_pred[i], threshold=THRESHOLD)
                if not classes:
                    classes = ["nocall"]

                row_id_lst.append(row.row_id)
                birds_lst.append(" ".join(classes))
        else:
            classes = []
            for pred in audio_pred:
                classes += pred2classes(pred, threshold=THRESHOLD)
            classes = list(set(classes))
            if not classes:
                classes = ["nocall"]

            row_id = group.row_id.unique()
            assert len(row_id) == 1
            row_id_lst.append(row_id[0])
            birds_lst.append(" ".join(classes))

    subm = pd.DataFrame({'row_id': row_id_lst, 'birds': birds_lst})
    subm.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    print("Experiments", EXPERIMENT)
    print("Device", DEVICE)
    print("Crop size", CROP_SIZE)
    print("Batch size", BATCH_SIZE)

    test_df = pd.read_csv(config.test_csv_path)
    audio_id2spec = prepare_audio_id_to_spec_data(test_df, config.audio)

    experiment_dir = config.experiments_dir / EXPERIMENT
    exp_pred = experiment_pred(experiment_dir, audio_id2spec)

    make_submission(test_df, exp_pred)

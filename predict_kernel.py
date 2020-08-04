import warnings
import pandas as pd

from src.predictor import Predictor
from src.transforms import get_transforms
from src.audio import read_audio, audio_to_melspectrogram

from src import config


warnings.filterwarnings('ignore',
                        'PySoundFile failed. Trying audioread instead.',
                        UserWarning)


CROP_SIZE = 320
BATCH_SIZE = 32
DEVICE = 'cuda'


def prepare_audio_id_to_spec_data(test_df, audio_params):
    audio_id2spec = dict()
    for audio_id in sorted(test_df.audio_id.unique()):
        file_path = config.test_audio_dir / (audio_id + '.mp3')
        audio, sr = read_audio(file_path, audio_params.sampling_rate)
        spec = audio_to_melspectrogram(audio, audio_params)
        audio_id2spec[audio_id] = spec
    return audio_id2spec


if __name__ == "__main__":
    test_df = pd.read_csv(config.test_csv_path)
    audio_id2spec = prepare_audio_id_to_spec_data(test_df, config.audio)

    transforms = get_transforms(False, CROP_SIZE)

    model_path = '/workdir/data/experiments/stride_002/fold_0/model-042-0.690526.pth'
    predictor = Predictor(model_path, transforms, BATCH_SIZE, CROP_SIZE, CROP_SIZE, DEVICE)

    spec = audio_id2spec['07ab324c602e4afab65ddbcc746c31b5']
    print(predictor.predict(spec).shape)

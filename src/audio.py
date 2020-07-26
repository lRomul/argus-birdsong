import warnings
import numpy as np

import librosa
import librosa.display

import torchaudio


warnings.filterwarnings('ignore',
                        'PySoundFile failed. Trying audioread instead.',
                        UserWarning)


def read_audio_librosa(file_path, sampling_rate):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    return y, sr


def read_audio_torchaudio(file_path, sampling_rate):
    y, sr = torchaudio.load(file_path, normalization=True)
    if sampling_rate != sr:
        y = torchaudio.transforms.Resample(sr, sampling_rate)(y)
    return y[0].cpu().numpy(), sampling_rate


def read_audio(file_path, sampling_rate):
    try:
        y, sr = read_audio_librosa(file_path, sampling_rate)
    except BaseException as e:
        print(f"Librosa load failed '{file_path}', try torchaudio")
        y, sr = read_audio_torchaudio(file_path, sampling_rate)
    return y, sr


def read_trim_audio(file_path, sampling_rate, min_seconds):
    min_samples = int(min_seconds * sampling_rate)
    try:
        y, sr = read_audio(file_path, sampling_rate)
        trim_y, trim_idx = librosa.effects.trim(y)  # trim, top_db=default(60)

        if len(trim_y) < min_samples:
            center = (trim_idx[1] - trim_idx[0]) // 2
            left_idx = max(0, center - min_samples // 2)
            right_idx = min(len(y), center + min_samples // 2)
            trim_y = y[left_idx:right_idx]

            if len(trim_y) < min_samples:
                padding = min_samples - len(trim_y)
                offset = padding // 2
                trim_y = np.pad(trim_y, (offset, padding - offset), 'constant')
        return trim_y
    except BaseException as e:
        print(f"Exception '{e}' while reading trim audio '{file_path}'")
        return np.zeros(min_samples, dtype=np.float32)


def audio_to_melspectrogram(audio, params):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=params.sampling_rate,
                                                 n_mels=params.n_mels,
                                                 hop_length=params.hop_length,
                                                 n_fft=params.n_fft,
                                                 fmin=params.fmin,
                                                 fmax=params.fmax,
                                                 power=params.power)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def show_melspectrogram(spectrogram, params, title='Log-frequency power spectrogram'):
    import matplotlib.pyplot as plt

    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel',
                             sr=params.sampling_rate,
                             hop_length=params.hop_length,
                             fmin=params.fmin,
                             fmax=params.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def read_as_melspectrogram(file_path, params, debug_display=False):
    audio = read_trim_audio(file_path, params.sampling_rate, params.min_seconds)
    spectrogram = audio_to_melspectrogram(audio, params)
    if debug_display:
        import IPython
        IPython.display.display(
            IPython.display.Audio(audio,
                                  rate=params.sampling_rate)
        )
        show_melspectrogram(spectrogram, params)
    return spectrogram

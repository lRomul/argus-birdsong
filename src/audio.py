import warnings
import numpy as np

import librosa
import librosa.display

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram


warnings.filterwarnings('ignore',
                        'PySoundFile failed. Trying audioread instead.',
                        UserWarning)


def read_audio_librosa(file_path, sampling_rate):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio, sr


def read_audio_torchaudio(file_path, sampling_rate):
    audio, sr = torchaudio.load(file_path, normalization=True)
    if sampling_rate != sr:
        audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio)
    return audio[0].cpu().numpy(), sampling_rate


def read_audio(file_path, sampling_rate):
    try:
        audio, sr = read_audio_librosa(file_path, sampling_rate)
    except BaseException as e:
        warnings.warn(f"Librosa load failed '{file_path}', '{e}', try torchaudio.")
        try:
            audio, sr = read_audio_torchaudio(file_path, sampling_rate)
        except BaseException as e:
            warnings.warn(f"Torchaudio load failed '{file_path}', '{e}', return zero array.")
            audio = np.zeros(sampling_rate, dtype=np.float32)
            audio, sr = audio, sampling_rate
    return audio, sr


def read_pad_audio(file_path, sampling_rate, min_seconds):
    min_samples = int(min_seconds * sampling_rate)
    try:
        audio, sr = read_audio(file_path, sampling_rate)

        if len(audio) < min_samples:
            padding = min_samples - len(audio)
            offset = padding // 2
            audio = np.pad(audio, (offset, padding - offset), 'constant')

    except BaseException as e:
        warnings.warn(f"Exception '{e}' while reading trim audio '{file_path}'.")
        audio = np.zeros(min_samples, dtype=np.float32)
        sr = sampling_rate

    return audio, sr


def audio_to_melspectrogram(audio, params):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=params.sampling_rate,
                                                 n_mels=params.n_mels,
                                                 hop_length=params.hop_length,
                                                 n_fft=params.n_fft,
                                                 fmin=params.fmin,
                                                 fmax=params.fmax,
                                                 power=params.power,
                                                 htk=params.htk,
                                                 norm=params.norm)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


@torch.no_grad()
def audio_to_melspectrogram_gpu(audio, params, device='cpu'):
    assert params.htk and params.norm is None
    device = torch.device(device)
    audio = torch.from_numpy(audio).to(device)
    melspect_transform = MelSpectrogram(sample_rate=params.sampling_rate,
                                        window_fn=torch.hann_window,
                                        hop_length=params.hop_length,
                                        n_mels=params.n_mels,
                                        n_fft=params.n_fft,
                                        f_min=params.fmin,
                                        f_max=params.fmax).to(device)
    spectrogram = melspect_transform(audio)
    power_to_db_transform = torchaudio.transforms.AmplitudeToDB('power', 80.)
    power_to_db_transform = power_to_db_transform.to(device)
    power_to_db_torch = power_to_db_transform(spectrogram)
    spectrogram = power_to_db_torch.squeeze().cpu().numpy()
    return spectrogram


def show_melspectrogram(spectrogram, params, figsize=(15, 3),
                        title='Log-frequency power spectrogram'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel',
                             sr=params.sampling_rate,
                             hop_length=params.hop_length,
                             fmin=params.fmin,
                             fmax=params.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def read_as_melspectrogram(file_path, params, pad=True, device='cpu', debug_display=False):
    if pad:
        audio, _ = read_pad_audio(file_path, params.sampling_rate, params.min_seconds)
    else:
        audio, _ = read_audio(file_path, params.sampling_rate)
    device = torch.device(device)
    if device.type == 'cuda':
        spectrogram = audio_to_melspectrogram_gpu(audio, params, device=device)
    else:
        spectrogram = audio_to_melspectrogram(audio, params)

    if debug_display:
        import IPython
        IPython.display.display(
            IPython.display.Audio(audio,
                                  rate=params.sampling_rate)
        )
        show_melspectrogram(spectrogram, params)
    return spectrogram

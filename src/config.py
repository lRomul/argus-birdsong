import os
import typing
from pathlib import Path
from pydantic import BaseModel


kernel_mode = False
if 'KERNEL_MODE' in os.environ and os.environ['KERNEL_MODE'] == 'predict':
    kernel_mode = True

if kernel_mode:
    input_data_dir = Path('/kaggle/input/birdsong-recognition/')
    if not (input_data_dir / 'test_audio').exists():
        input_data_dir = Path('/kaggle/input/birdcall-check/')
    output_data_dir = Path('/kaggle/working/data')
else:
    input_data_dir = Path('/workdir/data/')
    output_data_dir = Path('/workdir/data/')

train_csv_path = input_data_dir / 'train.csv'
test_csv_path = input_data_dir / 'test.csv'
train_audio_dir = input_data_dir / 'train_audio'
test_audio_dir = input_data_dir / 'test_audio'
example_test_audio_dir = input_data_dir / 'example_test_audio'
example_test_audio_metadata_csv_path = input_data_dir / 'example_test_audio_metadata.csv'
example_test_audio_summary_csv_path = input_data_dir / 'example_test_audio_summary.csv'
sample_submission_path = input_data_dir / 'sample_submission.csv'

train_folds_path = output_data_dir / 'train_folds_v1.csv'
experiments_dir = output_data_dir / 'experiments'
predictions_dir = output_data_dir / 'predictions'
prepared_train_dir = output_data_dir / 'prepared_train'

classes = [
    'aldfly', 'ameavo', 'amebit', 'amecro', 'amegfi', 'amekes', 'amepip', 'amered', 'amerob',
    'amewig', 'amewoo', 'amtspa', 'annhum', 'astfly', 'baisan', 'baleag', 'balori', 'banswa',
    'barswa', 'bawwar', 'belkin1', 'belspa2', 'bewwre', 'bkbcuc', 'bkbmag1', 'bkbwar', 'bkcchi',
    'bkchum', 'bkhgro', 'bkpwar', 'bktspa', 'blkpho', 'blugrb1', 'blujay', 'bnhcow', 'boboli',
    'bongul', 'brdowl', 'brebla', 'brespa', 'brncre', 'brnthr', 'brthum', 'brwhaw', 'btbwar',
    'btnwar', 'btywar', 'buffle', 'buggna', 'buhvir', 'bulori', 'bushti', 'buwtea', 'buwwar',
    'cacwre', 'calgul', 'calqua', 'camwar', 'cangoo', 'canwar', 'canwre', 'carwre', 'casfin',
    'caster1', 'casvir', 'cedwax', 'chispa', 'chiswi', 'chswar', 'chukar', 'clanut', 'cliswa',
    'comgol', 'comgra', 'comloo', 'commer', 'comnig', 'comrav', 'comred', 'comter', 'comyel',
    'coohaw', 'coshum', 'cowscj1', 'daejun', 'doccor', 'dowwoo', 'dusfly', 'eargre', 'easblu',
    'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eucdov', 'eursta', 'evegro', 'fiespa',
    'fiscro', 'foxspa', 'gadwal', 'gcrfin', 'gnttow', 'gnwtea', 'gockin', 'gocspa', 'goleag',
    'grbher3', 'grcfly', 'greegr', 'greroa', 'greyel', 'grhowl', 'grnher', 'grtgra', 'grycat',
    'gryfly', 'haiwoo', 'hamfly', 'hergul', 'herthr', 'hoomer', 'hoowar', 'horgre', 'horlar',
    'houfin', 'houspa', 'houwre', 'indbun', 'juntit1', 'killde', 'labwoo', 'larspa', 'lazbun',
    'leabit', 'leafly', 'leasan', 'lecthr', 'lesgol', 'lesnig', 'lesyel', 'lewwoo', 'linspa',
    'lobcur', 'lobdow', 'logshr', 'lotduc', 'louwat', 'macwar', 'magwar', 'mallar3', 'marwre',
    'merlin', 'moublu', 'mouchi', 'moudov', 'norcar', 'norfli', 'norhar2', 'normoc', 'norpar',
    'norpin', 'norsho', 'norwat', 'nrwswa', 'nutwoo', 'olsfly', 'orcwar', 'osprey', 'ovenbi1',
    'palwar', 'pasfly', 'pecsan', 'perfal', 'phaino', 'pibgre', 'pilwoo', 'pingro', 'pinjay',
    'pinsis', 'pinwar', 'plsvir', 'prawar', 'purfin', 'pygnut', 'rebmer', 'rebnut', 'rebsap',
    'rebwoo', 'redcro', 'redhea', 'reevir1', 'renpha', 'reshaw', 'rethaw', 'rewbla', 'ribgul',
    'rinduc', 'robgro', 'rocpig', 'rocwre', 'rthhum', 'ruckin', 'rudduc', 'rufgro', 'rufhum',
    'rusbla', 'sagspa1', 'sagthr', 'savspa', 'saypho', 'scatan', 'scoori', 'semplo', 'semsan',
    'sheowl', 'shshaw', 'snobun', 'snogoo', 'solsan', 'sonspa', 'sora', 'sposan', 'spotow',
    'stejay', 'swahaw', 'swaspa', 'swathr', 'treswa', 'truswa', 'tuftit', 'tunswa', 'veery',
    'vesspa', 'vigswa', 'warvir', 'wesblu', 'wesgre', 'weskin', 'wesmea', 'wessan', 'westan',
    'wewpew', 'whbnut', 'whcspa', 'whfibi', 'whtspa', 'whtswi', 'wilfly', 'wilsni1', 'wiltur',
    'winwre3', 'wlswar', 'wooduc', 'wooscj2', 'woothr', 'y00475', 'yebfly', 'yebsap', 'yehbla',
    'yelwar', 'yerwar', 'yetvir'
]

target2class = {trg: cls for trg, cls in enumerate(classes)}
class2target = {cls: trg for trg, cls in enumerate(classes)}
n_classes = len(classes)
n_folds = 5
folds = list(range(n_folds))


class AudioParams(BaseModel):
    sampling_rate: int
    hop_length: int
    fmin: int
    fmax: int
    n_mels: int
    n_fft: int
    power: float
    min_seconds: float
    htk: bool
    norm: typing.Any


audio = AudioParams(
    sampling_rate=44100,
    hop_length=690,
    fmin=20,
    fmax=22050,
    n_mels=128,
    n_fft=2560,
    power=2.0,
    min_seconds=2.0,
    htk=True,
    norm=None
)

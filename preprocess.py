import glob
from utils.display import *
from utils.dsp import *
from utils import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse
from utils.text.recipes import ljspeech
from utils.files import get_files
from pathlib import Path


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path_clean', '-c', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--path_noisy', '-n', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--path_clean_test', default=None, help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--path_noisy_test', default=None, help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path_clean is None:
    raise RuntimeError("Specify path_clean")
if args.path_noisy is None:
    raise RuntimeError("Specify path_noisy")

extension = args.extension
path_clean = args.path_clean
path_noisy = args.path_noisy
path_clean_test = args.path_clean_test
path_noisy_test = args.path_noisy_test

def convert_file(clean_path: Path, noisy_path: Path):
    clean = load_wav(clean_path)
    noisy = load_wav(noisy_path)
    peak = max(np.abs(clean).max(), np.abs(noisy).max())
    if hp.peak_norm or peak > 1.0:
        clean /= peak
        noisy /= peak
    mel = melspectrogram(noisy)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(clean, mu=2**hp.bits) if hp.mu_law else float_2_label(clean, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(clean, bits=16)
    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(input_paths):
    input_clean_path, input_noisy_path = input_paths
    print("Processing -> {0} vs {1}".format(input_clean_path, input_noisy_path))
    if input_clean_path.stem != input_noisy_path.stem:
        raise RuntimeError("Those are different samples!{0} vs {1}".format(input_clean_path, input_noisy_path))
    wav_id = input_clean_path.stem
    m, x = convert_file(input_clean_path, input_noisy_path)
    np.save(paths.mel/f'{wav_id}.npy', m, allow_pickle=False)
    np.save(paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]

def process_wav_test(input_paths):
    input_clean_test_path, input_noisy_test_path = input_paths
    print("Processing -> {0} vs {1}".format(input_clean_test_path, input_noisy_test_path))
    if input_clean_test_path.stem != input_noisy_test_path.stem:
        raise RuntimeError("Those are different samples!{0} vs {1}".format(input_clean_test_path, input_noisy_test_path))
    wav_id = input_clean_test_path.stem
    m, x = convert_file(input_clean_test_path, input_noisy_test_path)
    np.save(paths.test_mel/f'{wav_id}.npy', m, allow_pickle=False)
    np.save(paths.test_quant/f'{wav_id}.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]


wav_files_clean = sorted(get_files(path_clean, extension), key=lambda p:p.stem)
wav_files_noisy = sorted(get_files(path_noisy, extension), key=lambda p:p.stem)
wav_files_clean_test = sorted(get_files(path_clean_test, extension), key=lambda p:p.stem) if path_clean_test is not None else None
wav_files_noisy_test = sorted(get_files(path_noisy_test, extension), key=lambda p:p.stem) if path_noisy_test is not None else None
paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

print(f'\n{len(wav_files_clean)} {extension[1:]} files found in "{path_clean}"\n')

# test correctness of train set
if len(wav_files_clean) == 0 or len(wav_files_noisy) == 0:
    raise RuntimeError("Empty test set")
if len(wav_files_clean) != len(wav_files_noisy):
    raise RuntimeError("Different number of samples!")
for x, y in zip(wav_files_clean, wav_files_noisy):
    if x.stem != y.stem:
        raise RuntimeError("Lost sample!")

# test correctness of test set
if wav_files_clean_test is not None:
    if len(wav_files_clean_test) == 0 or len(wav_files_noisy_test) == 0:
        raise RuntimeError("Empty test set")
    if len(wav_files_clean_test) != len(wav_files_noisy_test):
        raise RuntimeError("Different number of samples!")
    for x, y in zip(wav_files_clean_test, wav_files_noisy_test):
        if x.stem != y.stem:
            raise RuntimeError("Lost sample!")
        
n_workers = max(1, args.num_workers)
simple_table([
    ('Sample Rate', hp.sample_rate),
    ('Bit Depth', hp.bits),
    ('Mu Law', hp.mu_law),
    ('Hop Length', hp.hop_length),
    ('CPU Usage', f'{n_workers}/{cpu_count()}')
])
pool = Pool(processes=n_workers)
dataset = []
wav_files = [(x, y) for x, y in zip(wav_files_clean, wav_files_noisy)]
for i, (item_id, length) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
    dataset += [(item_id, length)]
    bar = progbar(i, len(wav_files))
    message = f'{bar} {i}/{len(wav_files)} '
    stream(message)
with open(paths.data/'dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

if len(wav_files_clean_test) != 0 and len(wav_files_noisy_test) != 0:
    dataset_test = []
    wav_files_test = [(x, y) for x, y in zip(wav_files_clean_test, wav_files_noisy_test)]
    for i, (item_id, length) in enumerate(pool.imap_unordered(process_wav_test, wav_files_test), 1):
        dataset_test += [(item_id, length)]
        bar = progbar(i, len(wav_files_test))
        message = f'{bar} {i}/{len(wav_files_test)} '
        stream(message)
    with open(paths.data/'test_dataset.pkl', 'wb') as f:
        pickle.dump(dataset_test, f)

print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')

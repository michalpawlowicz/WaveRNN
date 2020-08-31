import librosa
import os
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', dest='dir', required=True, default=None, help='Resample dir')
parser.add_argument('--sr', dest='sr', required=True, type=int, help='val')

args = parser.parse_args()

SR=int(args.sr)

for wav_file in os.scandir(args.dir):
    if wav_file.path.endswith(".wav"):
        y, _ = librosa.load(wav_file, sr=SR)
        librosa.output.write_wav(wav_file, y, SR)
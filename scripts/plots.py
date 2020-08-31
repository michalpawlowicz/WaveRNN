import argparse
import matplotlib.pyplot as plt
import librosa
import librosa.display

import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--refpath', dest='refpath', required=True, help='Ref path or dir')
parser.add_argument('--degpath', dest='degpath', required=True, help='Deg path or dir')
parser.add_argument('--noisypath', dest='noisypath', required=True, help='Deg path or dir')
parser.add_argument('--savepath', dest='savepath', required=True, default=None, help='Save output to csv')

args = parser.parse_args()

y_ref, sr = librosa.load(args.refpath)
y_deg, sr = librosa.load(args.degpath)
y_noisy, sr = librosa.load(args.noisypath)

MAX_SR=22050

SIZE=(20, 15)

######### WAVEFORM ############
plt.figure(figsize=SIZE)
plt.subplot(3, 1, 1)
librosa.display.waveplot(y_noisy, sr=sr, max_sr=MAX_SR)
plt.title("Noisy waveform")
plt.subplot(3, 1, 2)
librosa.display.waveplot(y_deg, sr=sr, max_sr=MAX_SR)
plt.title("Enhanced waveform")
plt.subplot(3, 1, 3)
librosa.display.waveplot(y_ref, sr=sr, max_sr=MAX_SR)
plt.title("Reference waveform")
plt.savefig("{0}_waveform.png".format(args.savepath))
##############################

plt.cla()


########### MEL #############

n_fft = 2048
hop_length = 512
n_mels = 128


plt.figure(figsize=SIZE)
plt.subplot(3, 1, 1)
S = librosa.feature.melspectrogram(y_noisy, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Noisy melspectrogram")

plt.subplot(3, 1, 2)
S = librosa.feature.melspectrogram(y_deg, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Enhanced melspectrogram")

plt.subplot(3, 1, 3)
S = librosa.feature.melspectrogram(y_ref, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Reference melspectrogram")
plt.savefig("{0}_mel.png".format(args.savepath))
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
plt.title("Zaszumiony głos")
plt.subplot(3, 1, 2)
librosa.display.waveplot(y_deg, sr=sr, max_sr=MAX_SR)
plt.title("Odszumiony głos")
plt.subplot(3, 1, 3)
librosa.display.waveplot(y_ref, sr=sr, max_sr=MAX_SR)
plt.title("Czysty głos")
plt.savefig("{0}_waveform.png".format(args.savepath))
##############################

plt.cla()


OFFSET=50
LEN=int(sr/ 20)
y_noisy_short = y_noisy[OFFSET:OFFSET+LEN]
y_ref_short = y_ref[OFFSET:OFFSET+LEN]
y_enh_short = y_deg[OFFSET:OFFSET+LEN]
plt.figure(figsize=SIZE)
plt.subplot(2, 1, 1)
plt.plot(range(0, len(y_ref_short)), y_ref_short, label='Czysty głos')
plt.plot(range(0, len(y_noisy_short)), y_noisy_short, label='Zaszumiony głos')
plt.legend(loc="upper right")
plt.xlabel("Czas")
plt.subplot(2, 1, 2)
plt.plot(range(0, len(y_ref_short)), y_ref_short, label='Czysty głos')
plt.plot(range(0, len(y_enh_short)), y_enh_short, label='Odszumiony głos')
plt.legend(loc="upper right")
plt.xlabel("Czas")
plt.savefig("{0}_waveform2.png".format(args.savepath))

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
plt.title("Zaszumiony głos")

plt.subplot(3, 1, 2)
S = librosa.feature.melspectrogram(y_deg, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Odszumiony głos")

plt.subplot(3, 1, 3)
S = librosa.feature.melspectrogram(y_ref, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Czysty głos")
plt.savefig("{0}_mel.png".format(args.savepath))
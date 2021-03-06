from pesq import pesq
import sys
import librosa
import os
import argparse
import scipy
import numpy as np
import csv
import semetrics

def cal_pesq(refpath, degpath, sr=16000):
    if isinstance(refpath, str) and isinstance(degpath, str):
        ref, rate = librosa.load(refpath, sr=sr)
        deg, rate = librosa.load(degpath, sr=sr)
        return pesq(rate, ref, deg, 'wb'), pesq(rate, ref, deg, 'nb')
    elif isinstance(refpath, list) and isinstance(degpath, list):
        result = []
        for refp, degp in zip(refpath, degpath):
            ref, rate = librosa.load(refp, sr=sr)
            deg, rate = librosa.load(degp, sr=sr)
            result.append(pesq(rate, ref, deg, 'wb'), pesq(rate, ref, deg, 'nb'))
        return result
    else:
        print("Pass path to wav or to directory")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--refpath', dest='refpath', required=True, help='Ref path or dir')
parser.add_argument('--degpath', dest='degpath', required=True, help='Deg path or dir')
parser.add_argument('--savepath', dest='savepath', required=False, default=None, help='Save output to csv')

args = parser.parse_args()

if os.path.isdir(args.refpath) and os.path.isdir(args.degpath):
    refs = list(filter(lambda p: os.path.isfile(p) and p.path.endswith(".wav"), sorted(os.scandir(args.refpath), key=lambda p: os.path.basename(p.path))))
    degs = list(filter(lambda p: os.path.isfile(p) and p.path.endswith(".wav"), sorted(os.scandir(args.degpath), key=lambda p: os.path.basename(p.path))))
    if len(refs) != len(degs):
        print("Missing samples")
        sys.exit(1)
    for x, y in zip(refs, degs):
        if os.path.basename(x) != os.path.basename(y):
            print("Samples don't match")
            sys.exit(1)
    acc = np.zeros((len(refs), 2)) 
    for idx, (refpath, degpath) in enumerate(zip(refs, degs)):
        wide, narrow = cal_pesq(refpath.path, degpath.path)
        acc[idx,:] = [wide, narrow]
        print(os.path.basename(refpath), wide, narrow)
        #print("semetrics: ", semetrics.composite(refpath.path, degpath.path))
    print("Wide stats: ", scipy.stats.describe(acc[:,0]))
    print("Narrow stats: ", scipy.stats.describe(acc[:, 1]))

    if args.savepath is not None:
        with open(args.savepath, 'w+', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["sample", "wide", "narrow"])
            for sample, row in zip(refs, acc):
                writer.writerow([os.path.basename(sample), row[0], row[1]])
            writer.writerow(["avg", scipy.stats.describe(acc[:,0])[2], scipy.stats.describe(acc[:,1])[2]])
else:
    _, ext1 = os.path.splitext(args.refpath)
    _, ext2 = os.path.splitext(args.degpath)
    if ext1 != ".wav" or ext2 != ".wav":
        raise RuntimeError("Give me .wav files")
    wide, narrow = cal_pesq(args.refpath, args.degpath)
    print(wide, narrow)

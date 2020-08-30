from pesq import pesq
import sys
import librosa
import os
import argparse

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

args = parser.parse_args()

if os.path.isdir(args.refpath) and os.path.isdir(args.degpath):
    refs = sorted(os.scandir(args.refpath), key=lambda p: os.path.basename(p.path))
    degs = sorted(os.scandir(args.degpath), key=lambda p: os.path.basename(p.path))
    if len(refs) != len(degs):
        print("Missing samples")
        sys.exit(1)
    for x, y in zip(refs, degs):
        if os.path.basename(x) != os.path.basename(y):
            print("Samples don't match")
            sys.exit(1)
    for refpath, degpath in zip(refs, degs):
        print(os.path.basename(refpath), cal_pesq(refpath.path, degpath.path))
else:
    _, ext1 = os.path.splitext(args.refpath)
    _, ext2 = os.path.splitext(args.degpath)
    if ext1 != ".wav" or ext2 != ".wav":
        raise RuntimeError("Give me .wav files")
    print(cal_pesq(args.refpath, args.degpath))
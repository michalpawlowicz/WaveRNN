import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = []

for f in ['data/aws_model_raw_56_speakers_batch32_Adam_lr_1e-04-20200826-223011-tag-Epoch loss.csv',
        'data/azure_model_raw_56_speakers_batch32_Adam_lr_1e-04-20200828-121942-tag-Epoch loss.csv']:
    with open(f, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            s = row[0].split(',')
            if len(s) == 3:
                x = int(s[1])
                y = float(s[2])
                X.append((x, y))

X = sorted(X, key=lambda t: t[0])

X, y = zip(*X)

OFFSET=100
SIZE=(15, 10)

plt.figure(figsize=SIZE)
plt.subplot(2, 1, 1)
plt.xlabel("Epoch")
plt.ylabel("Entropia krzyżowa")
plt.title("Entropia krzyżowa dla każdego batcha")
df = pd.DataFrame(list(y))
plt.plot(df[0], 'lightblue', df[0].rolling(32).mean(), 'b')

plt.subplot(2, 1, 2)
df = pd.DataFrame(list(y)[OFFSET:])
plt.plot(df[0], 'lightblue', df[0].rolling(32).mean(), 'b')
plt.xlabel("Epoch")
plt.ylabel("Entropia krzyżowa")
plt.title("Entropia krzyżowa dla każdego batcha")
plt.savefig("checkpoints_loss_per_epoch.png")
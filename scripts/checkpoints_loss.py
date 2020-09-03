import matplotlib.pyplot as plt
import os
import re

CHECKPOINTS_DIR="/home/michal/Dokumenty/data/all_raw_checkpoints"
OFFSET=100
pattern = r".*epoch-(\d+)-loss-(\d+\.\d+).*"

X = []
for p in os.scandir(CHECKPOINTS_DIR):
    x = re.search(pattern, p.path)
    if x is not None:
        epoch, loss = x.groups()
        X.append((int(epoch), float(loss)))

X = sorted(X, key=lambda t: t[0])

X, y = zip(*X)

SIZE=(15, 10)
plt.figure(figsize=SIZE)
plt.subplot(2, 1, 1)
plt.xlabel("Epoch")
plt.ylabel("Entropia krzyżowa - koszt")
plt.title("Funkcja kosztu")
plt.plot(list(X), list(y))

plt.subplot(2, 1, 2)
plt.plot(list(X)[OFFSET:], list(y)[OFFSET:])
plt.xlabel("Epoch")
plt.ylabel("Entropia krzyżowa - koszt")
plt.title("Funkcja kosztu - przybliżone")
plt.savefig("checkpoints_loss.png")
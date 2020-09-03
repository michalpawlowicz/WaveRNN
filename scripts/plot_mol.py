import os
import csv
import matplotlib.pyplot as plt
import numpy as np

X_all = []
for f in sorted(os.scandir("data/mol/"), key=lambda p: os.path.basename(p.path)):
    X = []
    with open(f, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            s = row[0].split(',')
            if len(s) == 3:
                x = int(s[1])
                y = float(s[2])
                X.append((x, y))
    print("{0} from {1}".format(len(X), f))
    X_all.append(X)

SIZE=(15, 10)
plt.figure(figsize=SIZE)
plt.xlabel("Epoch")
plt.ylabel("Mixture of logistics")
plt.title("Mixture of logistics dla róznych hyper parametrów")
for D, label in zip(X_all, ["batch = 32, lr = 1e-4", "batch=64, lr=1e-4", " batch=128, lr=1e-4", "batch=32, lr=lr-4", "batch=32, lr=1e-4", "batch=32, lr=1e-4"]):
    X, y = zip(*D)
    plt.plot(X, y, label=label)

plt.legend(loc="upper right")
plt.savefig("mol_testing_hyper_params.png")
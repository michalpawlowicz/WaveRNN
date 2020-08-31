#!/bin/bash

python train_wavernn.py --lr 0.00001 --batch_size 32 --force_train --model_name model_raw_56_speakers_batch32_Adam_lr_1e-05 --adam --epoch 1102

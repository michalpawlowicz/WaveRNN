#!/bin/bash

find testinput/ -type f | xargs -I {} python gen_wavernn.py --voc_weights $1 --file {}  --force_cpu

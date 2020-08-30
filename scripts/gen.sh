#!/bin/bash

INPUT_DIR=$1
MODEL_CHECKPOINT=$2

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Pass input dir with test files in fist parameter"
    exit 1
fi

if [[ ! -f "$MODEL_CHECKPOINT" ]]; then
    echo "Pass path to model checkpoint as second parameter"
    exit 1
fi

find $INPUT_DIR -type f | xargs -I {} python gen_wavernn.py --voc_weights $MODEL_CHECKPOINT --file {} --force_cpu

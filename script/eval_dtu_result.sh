#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0"

DATASET="./dtu_data/eval_dtu" ##set to your SampleSet_MVS_Data
OUT_DIR='./ckpts/retr/outputs'
python dtu_eval.py --dataset_dir $DATASET $@ --outdir $OUT_DIR $@
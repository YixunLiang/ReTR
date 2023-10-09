#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

DATASET="./DTU_training/DTU_TEST" ##set to your DTU_TEST dataset direction 

LOAD_CKPT="./ckpts/retr/epoch=15-step=205808.ckpt"
OUT_DIR="./ckpts/retr/outputs"

python main.py --extract_geometry --set 0 --occ_trans --test_sample_coarse 64 --test_sample_fine 64 \
--test_n_view 3 --test_ray_num 4800 --volume_reso 128 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@ --debug \
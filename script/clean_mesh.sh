#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
TEST_DIR="./DTU_TEST" ##set to your DTU_TEST dataset direction 
EXP_DIR="./ckpts/retr/outputs/mesh"
python clean_mesh.py --root_dir $TEST_DIR $@ --out_dir $EXP_DIR $@ --n_view 3 --set 0 #--scan 24


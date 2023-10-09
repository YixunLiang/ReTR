#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

ROOT_DIR="./ckpts/retr/outputs"
python tsdf_fusion.py --n_view 3 --voxel_size 1.5 --margin 3 \
--root_dir=$ROOT_DIR $@



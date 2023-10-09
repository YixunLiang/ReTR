#!/usr/bin/env bash
DATASET="./DTU_training/mvs_training/dtu" ##set to your training dataset direction 
LOG_DIR="retr"
# Devices="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
python main.py --max_epochs 16 --batch_size 2 --occ_trans \
--weight_rgb 1.0 --weight_depth 1.0 \
--train_ray_num 1024 --volume_reso 128 \
--root_dir=$DATASET --exp_name=$LOG_DIR $@ \
--devices=$Devices --debug 
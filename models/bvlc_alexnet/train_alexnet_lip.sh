#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_lip.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_larc_B8k_lr0.5p2_wd0.0005_m0.9_0.95_100e_eta0.05_fp16.log

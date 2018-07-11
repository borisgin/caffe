#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_lip.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_larct_wd_B16k_lr3.5p2_wd0.0005_m0.9_0.96_eta0.064_fp16.log

# --snapshot=models/bvlc_alexnet/snapshots/alexnet_B16K_4000_iter_4000.solverstate \

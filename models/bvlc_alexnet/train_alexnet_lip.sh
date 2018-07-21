#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_lip.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_larc2t_B16k_lr5.0p2_wd0.0005_m0.9_eta0.005.log

# --snapshot=models/bvlc_alexnet/snapshots/alexnet_B16K_4000_iter_4000.solverstate \

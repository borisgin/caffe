#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/cifar10/solver_bn.prototxt -gpu=0 \
  2>&1 | tee examples/cifar10/cifar10_bn_lr0.01.log


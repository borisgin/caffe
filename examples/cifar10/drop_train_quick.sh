#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/drop_cifar10_quick_solver.prototxt 2>&1 | tee examples/cifar10/logs/drop_log.txt

# reduce learning rate by factor of 10 after 8 epochs
#$TOOLS/caffe train \
#  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
#  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate

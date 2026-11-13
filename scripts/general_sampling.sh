#!/usr/bin/env bash

# Replace these placeholders with the directories where samples should be saved.

python general_sampling.py \
    -dd ../simulated_data \
    -bc example_config.yaml \
    -niter 25 \
    -cs ../chains/test/scripttest.npz

#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2


python test_ood.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir cifar10 \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_path checkpoints/pretrained_models/cifar10_wrn_pretrained_epoch_99.pt \
--batch 128 \
--logdir checkpoints/test_log \
--score ${METHOD}

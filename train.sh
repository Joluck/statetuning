#!/bin/bash

cd "$(dirname "$0")"

python train.py \
    --load_model /home/rwkv/models/rwkv7/rwkv7-g1a-0.4b-20250905-ctx4096.pth \
    --data_path /home/rwkv/datas/bad_lan.jsonl \
    --output_dir ./outmodel \
    --vocab_size 65536 \
    --n_embd 1024 \
    --n_layer 24 \
    --precision bf16 \
    --batch_size 4 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --ctx_len 2048

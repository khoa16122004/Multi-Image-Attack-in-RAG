#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python main.py \
    --sample_path run.txt \
    --result_clean_dir result_usingquery=0_clip \
    --reader_name llava-one \
    --retriever_name clip \
    --pop_size 20  \
    --F 0.9 \
    --n_k 5 \
    --max_iter 100 \
    --std 0.05 \
    --using_question 1


CUDA_VISIBLE_DEVICES=5 python main.py \
    --sample_path run.txt \
    --result_clean_dir result_usingquery=0_clip \
    --reader_name llava-next \
    --retriever_name clip \
    --pop_size 20  \
    --F 0.9 \
    --n_k 5 \
    --max_iter 100 \
    --std 0.05 \
    --using_question 1


CUDA_VISIBLE_DEVICES=7 python main.py \
    --sample_path run.txt \
    --result_clean_dir result_usingquery=0_clip \
    --reader_name qwenvl2.5 \
    --retriever_name clip \
    --pop_size 20  \
    --F 0.9 \
    --n_k 5 \
    --max_iter 100 \
    --std 0.05 \
    --using_question 1

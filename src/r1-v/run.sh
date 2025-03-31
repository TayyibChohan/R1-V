#!/bin/bash

export DEBUG_MODE="true"
export LOG_PATH="./debug1epoch_run.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12445" \
    src/open_r1/grpo.py \
    --output_dir /data/tayyibc/r1-v_out \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name AI4Math/MathVista \
    --deepspeed ./zero3.json \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-7B-GRPO-MathVista-5k \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-6 \
    --warmup_steps 50 \
    --max_steps 4000 \
    --num_generations 4   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

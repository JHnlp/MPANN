#! /bin/sh

#CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 run_mpann.py \
CUDA_VISIBLE_DEVICES=0 python run_mpann.py \
  --data_dir=./data/bioasq2021 \
  --mesh_file=./data/MeSH/mesh_headings_2021.jsonl \
  --output_dir=./output/_bioasq_2021_test9a_ \
  --model_type=multiprobe \
  --tokenizer_name=./model/bioasq2021_shared_base/checkpoint-250000 \
  --model_name_or_path=./model/bioasq2021_shared_base/checkpoint-250000/pytorch_model.bin \
  --config_name=./model/bioasq2021_shared_base/checkpoint-250000/config.json \
  --train_file=./data/bioasq2021/additional/sample_docs_train.jsonl \
  --dev_file=./data/bioasq2021/additional/sample_dev.jsonl \
  --test_file=./data/bioasq2021/test_filtered/Task9a-Batch1-Week1_raw_filtered.jsonl \
  --log_file=./_bioasq_task9a.txt \
  --task_name=bioasq \
  --do_test \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_output_dir \
  --max_seq_length=512 \
  --per_gpu_train_batch_size=2 \
  --per_gpu_eval_batch_size=20 \
  --learning_rate=1e-5 \
  --num_train_epochs=2.0 \
  --logging_steps=20 \
  --save_steps=2000 \
  --weight_decay=1e-5 \
  --warmup_proportion=0.1 \
  --max_step=80000 \
  --gradient_accumulation_steps=10
#  --overwrite_cache

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/IdiomAll \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir models/IdiomAll\
  --overwrite_output_dir


python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/MRPC \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir models/MRPC \
  --overwrite_output_dir

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/IdiomDaniel \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir models/IdiomDaniel \
  --overwrite_output_dir

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/IdiomElena \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir models/IdiomElena \
  --overwrite_output_dir

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/TrainMRPCTestOurs \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir models/TrainMRPCTestOurs \
  --overwrite_output_dir


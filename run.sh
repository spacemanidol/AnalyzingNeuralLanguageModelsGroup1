#!/bin/bash
for i in {1..18}
do
  python3 run_glue.py \
  --data_dir data/Exp$i\
  --num_train_epochs 10.0 \
  --output_dir models/Exp$i 
done
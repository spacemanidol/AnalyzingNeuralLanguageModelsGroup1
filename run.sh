python3 run_glue.py \
  --data_dir data/FullSample/TrainTestMRPC5800samples80:20\
  --num_train_epochs 10.0 \
  --output_dir models/TrainTestMRPC5800samples80:20

python3 run_glue.py \
  --data_dir data/FullSample/TrainTestOurDataIdiomsInDevNotInTrain2000Samples80:20\
  --num_train_epochs 10.0 \
  --output_dir models/TrainTestOurDataIdiomsInDevNotInTrain2000Samples80:20

python3 run_glue.py \
  --data_dir data/FullSample/TrainTestOurDataRandomSample2000Samples80:20\
  --num_train_epochs 10.0 \
  --output_dir models/TrainTestOurDataRandomSample2000Samples80:20

python3 run_glue.py \
  --data_dir data/1kSample/TrainTestDanielRandomSample1000Samples \
  --num_train_epochs 10.0 \
  --output_dir models/TrainTestDanielRandomSample1000Samples

python3 run_glue.py \
  --data_dir data/1kSample/TrainTestElenaRandomSample1000Samples \
  --num_train_epochs 10.0 \
  --output_dir models/TrainTestElenaRandomSample1000Samples

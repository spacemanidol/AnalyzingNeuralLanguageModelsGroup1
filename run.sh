echo "Training and Testing BERT"
python3 bert.py --data_dir data/cola --train --eval --epochs 1
echo "Training and Testing BiLSTM"
python3 bilstm.py --data_dir data/cola --train --eval --epochs 1

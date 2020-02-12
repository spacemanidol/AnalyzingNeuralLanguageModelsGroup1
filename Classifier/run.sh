echo "Training and Testing Bilstm"
python model.py --data_dir $1/cola_formated --train --evaluate --model bilstm
echo "Training and Testing Bert"
python model.py --data_dir $1/cola_formated --train --evaluate --model bert
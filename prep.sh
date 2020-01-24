cd data
wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
unzip cola_public_1.1.zip
rm cola_public_1.1.zip
cd ..
python modify_format.py data/cola_public/raw/in_domain_train.tsv data/cola/train.tsv
python modify_format.py data/cola_public/raw/in_domain_dev.tsv data/cola/dev.tsv
python modify_format.py data/cola_public/raw/out_of_domain_dev.tsv data/cola/out_dev.tsv
rm -rf data/cola_public
sudo pip3 install -r requirements.txt
./run.sh

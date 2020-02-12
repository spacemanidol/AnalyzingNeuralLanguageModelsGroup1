wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
unzip cola_public_1.1.zip
mkdir cola_formated
python modify_format.py cola_public/raw/in_domain_train.tsv cola_formated/train.tsv
python modify_format.py cola_public/raw/in_domain_dev.tsv cola_formated/dev.tsv
python modify_format.py cola_public/raw/out_of_domain_dev.tsv cola_formated/out_dev.tsv
rm cola_public_1.1.zip
rm -rf cola_public
mv cola_formated $1
./run.sh $1
python -m spacy download en
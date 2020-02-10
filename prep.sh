cd data
wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
unzip cola_public_1.1.zip
wget https://files.pushshift.io/reddit/comments/RC_2011-08.bz2
bz2 -d RC_2011-08.bz2
cd ..
python modify_format.py data/cola_public/raw/in_domain_train.tsv data/cola/train.tsv
python modify_format.py data/cola_public/raw/in_domain_dev.tsv data/cola/dev.tsv
python modify_format.py data/cola_public/raw/out_of_domain_dev.tsv data/cola/out_dev.tsv
python get_idiom_comments.py data/idiomLexicon.tsv data/RC_2011-08 data/idiomcomments 5
rm data/RC_2011-08.bz2
rm data/cola_public_1.1.zio
rm -rf data/cola_public
pip3 install --user -r requirements.txt

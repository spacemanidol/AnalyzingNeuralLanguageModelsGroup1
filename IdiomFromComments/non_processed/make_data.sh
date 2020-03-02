python make_unified.py 
shuf all.tsv > all_shuffled.tsv 
#3000 all_shuffled.tsv
tail -n 600 all_shuffled.tsv >> dev.tsv
head -n 2400 all_shuffled.tsv >> train.tsv
mv dev.tsv ../../data/
mv train.tsv ../../data/
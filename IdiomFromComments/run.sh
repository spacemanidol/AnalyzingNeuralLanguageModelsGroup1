wget https://files.pushshift.io/reddit/comments/RC_2011-08.bz2
bz2 -d RC_2011-08.bz2
python get_idiom_comments.py {$1}/idiomLexicon.tsv RC_2011-08 {$1}/idiomcomments.tsv 5
rm data/RC_2011-08.bz2
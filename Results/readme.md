This folder holds details for experiments starting with a finetuned bert model, then training a linear classifier on top (using run_probe.py). 
Parameters of the embedding model (variation of BERT) were held static during training, while the parameters of the linear classifier were trained

--------------------
description of files
--------------------
dataset_description.tsv - description of the datasets used in experimentation

model_description.tsv - description of each pretrained model used in experimentation

pretrain_experiment_description - this does not directly apply to results in this folder. This is a description of the experiments used to produced the "pretrained" models (distilled from Daniels readme elsewhere)

result_full_detail.tsv - (hopefully) all relevant details for each trial, organized by trial. This should be enough to reproduce any result.

result_summary_best_scores.tsv - more concise summary of trial results, only on the test (dev) dataset. Many trials used all of the same details except for embedding paradigm. This file reports the best scores (accuracy, f1, precision, recall) for either the cls or combined paradigm, all other things equal.

build_summary.py - a script to build a summary of each individual test_classifications.tsv file, which is produced by run_probe.py. Summary includes accuracy, f1 score, precision, recall, and a count of true_pos, false_pos, true_neg, false_neg

build_all_summaries.sh - a script that applies the build_summary.py script to all subfolders in the pwd

-------------------------
description of subfolders
-------------------------
Each subfolder was copied from the "output" folder of a given trial, which is produced by the script run_probe.py. 

There are a few standard files within each subfolder:
    - test_classifications.tsv - the output of run_probe.py. Detailed log of judgements for each sentence,paraphrase pair
    - summary_xxx_xxx - the output of build_summary.py. Contains accuracy, f1 score, precision, recall, and a count of true_pos, false_pos, true_neg, false_neg
    - test_acc.txt - the accuracy of the given trial, output by run_probe.py. NOTE: accuracy may be slightly off due to presence of -1 and 2 for classifier judgments

Additionally, the subfolders corresponding to training runs have more data corresponding to the output of run_probe.py

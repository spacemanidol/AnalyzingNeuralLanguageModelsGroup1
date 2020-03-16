# AnalyzingNeuralLanguageModelsGroup1
Group 1 for LING 575: Analyzing Neural Language Models [Win '20] 

By Daniel Campos, Paige Finkelstein, Elena Khasanova, Wes Rose, and Josh Tanner

### Summary

This project aims to explore BERT's understanding of idioms and their paraphrases. More concretely, we aim to:
(1) Test BERT's ability to do paraphrase classification, where one of the sentences contains an idiom.
(2) Expore the similarity of the embedding vectors produced by BERT where one of the vectors is related to an idiom (either contains an idiom, exactly represents an idiom, or represents a substring of an idiom).

The primary directories to use for reproducing and expanding on our experiments are the probe directory and the vector_similarity directory.

### Vector Similarity Experiments 
See the [README.md](https://github.com/spacemanidol/AnalyzingNeuralLanguageModelsGroup1/tree/master/vector_similarity) in the `vector_similarity` directory for detailed instructions on the data and how to run the experiments.

### Idiom Paraphrase Detection Probing Classifier Experiments

The probe directory contains the script run_probe.py, which can be used to run paraphrase classification experiments. run_probe.py takes many arguments. At the highest level, it can be used to train our linear classifier or test our linear classifier. Following is a description of arguments, and an indicator of whether the argument is used for training, testing, or both.

--run: "train" or "test". self-explanatory\
--input: path to the data. required for train and test.\
--run_name: label for the run, which will be used for output directories. defaults to run_[timestamp]. Use for train and test.\
--indices: the indices for the label, sentence 1, and sentence 2 in your data file. Defaults to 0.3.4. Use for train and test.\
--embedding_batch_size: The size of the batch for generating BERT embeddings. Defaults to 64. Use for train and test.\
--embedding_cache: The path to the cached BERT embeddings. Optional. Only use if data, paradigm, and embedding_batch_size are unchanged.\
--embedding_model: Identifier for the underlying BERT model to use to generate embeddings. Either use one from the Transformers library or your own (or one from our models folder). Required for train and test.\
--embedding_paradigm: The method for generating a single vector from BERT embeddings for sentence pairs. cls, combined, or cls_pool. Required for train and test.\
--model: Name for the saved probe model to load. Required only for test.\
--learning_rate: learning rate for training the linear classifier. Required only for train.\
--epochs: number of epochs to use for training the linear classifier. Required only for train.\
--min_loss_step: If the improvement between epochs is less than this number, training will stop early. Required only for train.\
--rand_seed: random seed. Default is 0. Use train and test.

Example command for training:

python3 /AnalyzingNeuralLanguageModelsGroup1/probe/run_probe.py --run train --input /AnalyzingNeuralLanguageModelsGroup1/data/t1.tsv --run_name trial_train --indices 0.3.4 --embedding_batch_size 64 --embedding_model /AnalyzingNeuralLanguageModelsGroup1/models/bert-base-uncased --embedding_paradigm cls --learning_rate 0.00247875217 --epochs 10000 --min_loss_step 0 --rand_seed 1583398837989

Example command for testing:

python3 /AnalyzingNeuralLanguageModelsGroup1/probe/run_probe.py --run test --input /AnalyzingNeuralLanguageModelsGroup1/data/d1.tsv --run_name trial_test --indices 0.3.4 --embedding_batch_size 64 --embedding_model /AnalyzingNeuralLanguageModelsGroup1/models/bert-base-uncased --embedding_paradigm cls --model output/trial_train/model.pt

### Data Folder

Data folder contains all of the data that we used for our experiments, as well as results for some finetuning experiments. The key for the data files is as follows:

t1.tsv: Full MRPC 4076/1726. Training Split\
t1,.tsv: MRPC 1600/400. Training Split\
t1,,.tsv: MRPC 800/200. Training Split\
t2.tsv: Our Idiom Data Random Split 1600/400. Training Split\
t2,.tsv: Our Idiom Data Random Split 800/200. Training Split\
t3.tsv: Our Idiom data disjunct 1600/400. Training Split\
t3,.tsv: Our Idiom data disjunct 800/200. Training Split\
t4.tsv: Our Idiom Data Method 1. Training Split\
t5.tsv: Our Idiom Data Method 2. Training Split\
d1.tsv: Full MRPC 4076/1726. Dev Split\
d1,.tsv: MRPC 1600/400. Dev Split\
d1,,.tsv: MRPC 800/200. Dev Split\
d2.tsv: Our Idiom Data Random Split 1600/400. Dev Split\
d2,.tsv: Our Idiom Data Random Split 800/200. Dev Split\
d3.tsv: Our Idiom data disjunct 1600/400. Dev Split\
d3,.tsv: Our Idiom data disjunct 800/200. Dev Split\
d4.tsv: Our Idiom Data Method 1. Dev Split\
d5.tsv: Our Idiom Data Method 2. Dev Split\

### Other Information

We used comments from Reddit. Raw data from [Reddit Comment Dataset](https://files.pushshift.io/reddit/comments/)

To help generate paraphrases, we used Idiom Lexicon downloaded from [IBM Debater Datasets](https://www.bing.com/search?q=project+debater+datasets&PC=U316&FORM=CHROMN)

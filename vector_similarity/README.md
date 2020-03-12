## Vector Similarity Comparisons

This module implements code for comparing similarity between deeply contextualized word embeddings. There are two main types of comparisons/calculations handled: 

1.) **Word-level embeddings** that compare differences of figurative words used in the context of idioms with three other classes of words (the same word used in literal contexts, a paraphrase of the contextual meaning of the idiomatic word, and completely random words).

2.) **Sentence-level embeddings** that are designed to examine the classification output results of our paraphrase detection linear classifier (or another model) and compare the similarity of sentence-level embeddings across the categories of results outcome (i.e. true positive, false positive, true negative, false negative) from the classifier.

For a visual example of the usage and output, see the jupyter notebook file in this directory: `run_vector_similarity_different_models_example.ipynb`

### 1. Word-level embeddings 

This model takes as input a data file with tab-separated fields as follows:

`sentence_id	pair_id	sentence	word	figurative`

The data is grouped into sections by `pair_id` (this name is somewhat misleading because it actually refers to many sentences that make up one group focused around a single idiom phrase. 

In this group, some of the sentences will be denoted as figurative uses (i.e. the value for `figurative` is `1`), and these sentences will all contain the idiom phrase. From this idiom phrase, there will be one word (`word`) denoted from the idiom. A second section of sentences in the `pair_id` group will be "literal" sentences, that use this same word in a *literal* way. These sentences have the same `pair_id` and the same `word` value, but their `figurative` value is `0`. Finally, there is a third section in the `pair_id` group that is made up of sentences that include a *paraphrase* word for the idiom word used in a literal way. These sentences have the same `pair_id` but a different `word` value (i.e. the paraphrased word) and a `figurative` value of `0`.

Here's an abbreviated example for the idiom phrase "on the other hand":
`sentence_id	 pair_id	 sentence 	 word	 figurative`

`122	270	On the other hand, it is kind of distracting.	hand	1`

`123	270	Samsung's hardware pioneering, on the other hand, needs more work.	hand	1`

`127	270	When the mom let go of the girl's hand, the girl sprang up and ran again to the stage.	hand	0`

`129	270	Grab his hand!	hand	0`

`130	270	I count 11 scars on this hand.	hand	0`

`142	270	This is a valid perspective on human alienation.	perspective	0`

`139	270	From my perspective as someone who has been an employer, I would say this.	perspective	0`

`140	270	I think that most people that have seen the world will agree with your perspective.	perspective	0`

There are also some sets of `pair_id` sentences that are all literal, each based on a single entirely random word, that get used as a control group of sorts in the calculations.

#### Purpose and Output
For each `pair_id` group based on an idiom phrase, we calculate the average cosine similarity and average euclidean distance between the different groupings. That is, we calculate an average score for the distance between *literal and literal* usages, between *figurative and figurative* usages, between *paraphrase and literal* usages, and between *paraphrase and figurative* usages. We also calculate the average distance between *figurative and random* as a control.

We also compute the average *differences* between these scores over all of the groups, to try to ascertain patterns between the cosine similarity and euclidean distance in general.

Finally, we also produce PCA visualizations for each `pair_id` group. We produce these PCA comparisons on three different sets from the data: 

1.) figurative (the word used in the idiom context) and literal uses of the idiom word

2.) figurative and literal (as in #1) plus the paraphrase usages with the same meaning as the idiom word in context

3.) same as those in #2 plus embeddings for a randomly chosen word

All of the results are stored in a folder with the `run_name` in the `output` directory in this current directory (`vector_similarity`). The PCA visualizations can be found in the run output folder under `PCA_images`.

#### Example usage

The first time you run, you will not provide an argument for `embedding_cache` and the embeddings will be generated from scratch and stored for future use.

`python3 run_vector_similarity.py --comparison_type words --input word_vec_sim_test.txt --embedding_model bert-large-uncased --run_name word_large_1 --embedding_batch_size 32`

After the first run, you can provide the `embedding_cache` argument with a string to the appropriate file:

`python3 run_vector_similarity.py --comparison_type words --input word_vec_sim_test.txt --embedding_model bert-large-uncased --run_name word_large_2 --embedding_batch_size 32 --embedding_cache cache/word_large_1`

For detailed explanation of the command line arguments, see the **Command line arguments** section below.

### 2. Sentence-level embeddings 

UPDATE: Most of this functionality has been directly incorporated into the probe directory as part of the probe run itself (run_probe.py) to ensure a match between the sentence embeddings used in the classification and those compared for vector similarity. However, one can still run this evaluation on the outputted classification file directly from here too as follows:

`python3 run_vector_similarity.py --comparison_type para_pairs --input ../probe/output/bert_large_combined_test/test_classifications.tsv --embedding_model bert-large-uncased --run_name sentence_word_large --embedding_batch_size 32`

#### Command line arguments

**comparison_type**: one of `words` or `para_pairs`. This determines whether you are doing the word-level or sentence-level comparison, and the data will need to correctly match the expected input format.

**input**: the path to the file containing the input data. See descriptions above for the necessary format for each comparison type.

**embedding_batch_size**: determines the size of the batches for when generating embeddings; multiples for 16 are common

**embedding_model**:  this is the model used to generate the embeddings. It can be any of the models provided by the Transformers library, or the path to a model you have saved locally.

**run_name**: determines the name of the run, i.e. how everything gets labeled and saved.

**embedding_cache**: path/name of a previous run that has already generated and cached the embeddings; will compute from scratch and save if not provided.

**show_pca**: for the word-level comparison only. If `True`, this will pop open the PCA plots rather than save the images (good for when running in jupyter notebook)

  

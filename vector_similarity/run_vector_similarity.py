import sys
sys.path.append('../')
from transformers import BertTokenizer, BertModel
import torch
import itertools
from collections import defaultdict
from probe.load_data import WordInspectionDataset, SentenceParaphraseInspectionDataset
from scipy.spatial.distance import cosine
from statistics import mean 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import argparse
import time

combined, cls = 'combined', 'cls'
words, paraphrase_sent_pairs = 'words', 'para_pairs'

def main(input_args):
    if input_args.comparison_type == words:
        word_usage_comparisons(input_args)
    else:
        sentence_paraphrase_comparisons(input_args)


def sentence_paraphrase_comparisons(input_args):
    dataset = SentenceParaphraseInspectionDataset(input_args.input, input_args.embedding_model, 
                                                input_args.embedding_batch_size, input_args.run_name)

    embeddings = get_embeddings(dataset, input_args.embedding_cache)
    embedding_outputs, encoded_inputs, _indices = embeddings

    sentence_embeddings = get_sentence_embeddings(embeddings, dataset, input_args.embedding_paradigm)
    paraphrase_cosine_metrics = calculate_sentence_paraphrase_cosine_metrics(dataset, embedding_outputs, 
                                                                            encoded_inputs, sentence_embeddings)
    
    print(paraphrase_cosine_metrics)
    print(summarize_sentence_similarity_comp(paraphrase_cosine_metrics))


def word_usage_comparisons(input_args):
    dataset = WordInspectionDataset(input_args.input, input_args.embedding_model, 
                                    input_args.embedding_batch_size, input_args.run_name)
    embeddings = get_embeddings(dataset, input_args.embedding_cache)
    embedding_outputs, encoded_inputs, _indices = embeddings
    data = dataset.get_data()
    idiom_sentence_indexes = get_idiom_sentences(data)

    word_sim_results = calculate_word_cosine_metrics(dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes)
    print(word_sim_results)
    print(summarize_word_similarity_comp(word_sim_results))

    PCA_comparisions(dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes)


def get_embeddings(data, embedding_cache):
    if embedding_cache is None:
        encoded_data = data.get_encoded()
        return data.bert_word_embeddings(encoded_data)
    return data.load_saved_embeddings(embedding_cache)

def get_sentence_embeddings(embeddings, data, embedding_paradigm):
    embedding_outputs, encoded_inputs, indices = embeddings
    if embedding_paradigm == combined:
        return data.aggregate_sentence_embeddings(embedding_outputs, encoded_inputs, indices)
    return data.bert_cls_embeddings(embeddings)

def calculate_sentence_paraphrase_cosine_metrics(dataset, embedding_outputs, encoded_inputs, sentence_embeddings):
    data = dataset.get_data()
    paraphrase_pairs = get_paraphrase_pairs(data)
    paraphrase_cosine_metrics = [calculate_paraphrase_pair_similarity(pair_sents, dataset, data, embedding_outputs, encoded_inputs, sentence_embeddings) 
                            for pair_sents in paraphrase_pairs]
    return paraphrase_cosine_metrics

def calculate_word_cosine_metrics(dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes):
    word_cosine_metrics = [calculate_word_similarity_metrics(idiom_sent_idx, dataset, embedding_outputs, encoded_inputs) 
                            for idiom_sent_idx in idiom_sentence_indexes]
    return word_cosine_metrics

def get_paraphrase_pairs(dataset):
    paraphrase_pairs = []
    for i, sent in enumerate(dataset):
        paraphrase_pairs.append((i, sent))
    return paraphrase_pairs

def calculate_paraphrase_pair_similarity(pair, dataset, data, embedding_outputs, encoded_inputs, sentence_embeddings):
    sent_1 = pair[0]
    sent_2 = pair[1]
    sent_1_index = sent_1[0]
    sent_2_index = sent_2[0]
    cosine_sim = 1 - cosine(sentence_embeddings[sent_1_index], sentence_embeddings[sent_2_index])
    
    return {
        'pair_id': sent_1[1].pair_id,
        'sent_1': dataset.decode(encoded_inputs[sent_1_index].tolist()),
        'sent_2': dataset.decode(encoded_inputs[sent_2_index].tolist()),
        'paraphrase': data[sent_1_index].true_label,
        'judgment': data[sent_1_index].classifier_judgment,
        'cosine_similarity': cosine_sim
    }    

def calculate_word_similarity_metrics(idiom_sent_index, dataset, embedding_outputs, encoded_inputs):
    data = dataset.get_data()
    idiom_ex = data[idiom_sent_index]
    idiom_word_embedding = get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, idiom_sent_index)
    cosine_similarity_metrics = {}

    literal_usage_sents = [i for i, ex in enumerate(data) if ex.pair_id == idiom_ex.pair_id and 
                                                            ex.word == idiom_ex.word and not 
                                                            ex.sentence_id == idiom_ex.sentence_id ]
    paraphrase_sents = [i for i, ex in enumerate(data) if ex.pair_id == idiom_ex.pair_id 
                                                            and not ex.word == idiom_ex.word]

    literal_usage_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, lit_idx) for lit_idx in literal_usage_sents]
    paraphrase_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, para_idx) for para_idx in paraphrase_sents]

    cosine_similarity_metrics['fig_to_literal'] = calculate_cosine_similarity_average([idiom_word_embedding], literal_usage_embeddings)
    cosine_similarity_metrics['literal_to_literal'] = calculate_cosine_similarity_average(literal_usage_embeddings)
    cosine_similarity_metrics['fig_to_paraphrase'] = calculate_cosine_similarity_average([idiom_word_embedding], paraphrase_embeddings)
    cosine_similarity_metrics['literal_to_paraphrase'] = calculate_cosine_similarity_average(literal_usage_embeddings, paraphrase_embeddings)
    
    return {
        'sentence_id': idiom_ex.sentence_id,
        'sentence': dataset.decode(encoded_inputs[idiom_sent_index].tolist()),
        'word': idiom_ex.word,
        'paraphrase_word': data[paraphrase_sents[0]].word,
        'cosine_similarities': cosine_similarity_metrics,
    }

def get_idiom_sentences(dataset):
    return [i for i, ex in enumerate(dataset) if ex.figurative]

def get_sentences_for_idiom_sentence(dataset, idiom_sent):
    literal_usage_sents = [i for i, ex in enumerate(dataset) if ex.pair_id == idiom_sent.pair_id and 
                                                                ex.word == idiom_sent.word and not 
                                                                ex.sentence_id == idiom_sent.sentence_id ]
    paraphrase_sents = [i for i, ex in enumerate(dataset) if ex.pair_id == idiom_sent.pair_id and not 
                                                            ex.word == idiom_sent.word]
    return (literal_usage_sents, paraphrase_sents)

def get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, dataset_index):
    ex = data[dataset_index]
    decoded_tokens = dataset.get_decoded_tokens(encoded_inputs[dataset_index].tolist())
    word_index = decoded_tokens.index(ex.word[0])
    return embedding_outputs[dataset_index][word_index]

def calculate_cosine_similarity_average(embeddings_1, embeddings_2=None):
    if embeddings_2:
        embedding_pairs = list(itertools.product(embeddings_1, embeddings_2))
    else:
        embedding_pairs = list(itertools.combinations(embeddings_1, 2))

    cosine_similarities = [1 - cosine(embedding_1, embedding_2) for embedding_1, embedding_2 in embedding_pairs]
    return mean(cosine_similarities)

# This computes the average difference in cosine similarity between:
# 1.) literal to literal usages versus figurative to literal usage
# 2.) figurative to paraphrase usages versus literal to paraphrase useage
def summarize_word_similarity_comp(results):
    literal_sim_advantage = [result['cosine_similarities']['literal_to_literal'] - result['cosine_similarities']['fig_to_literal'] for result in results]
    fig_to_paraphrase_advantage = [result['cosine_similarities']['fig_to_paraphrase'] - result['cosine_similarities']['literal_to_paraphrase'] for result in results]
    
    summary_stats = {
        'lit_to_lit_improvement_over_fig_to_lit': mean(literal_sim_advantage),
        'fig_to_paraphrase_improvement_over_lit_to_paraphrase': mean(fig_to_paraphrase_advantage)
    }
    return summary_stats

# This computes the average cosine similarity scores between paraphrase pairs,
# grouped into 4 categories based on gold label (i.e. true paraphrase or not) and classifier judgment
def summarize_sentence_similarity_comp(results):
    correctly_judged_paraphrases = [result['cosine_similarity'] for result in results if result['paraphrase'] and result['judgment']]
    correctly_judged_non_paraphrases = [result['cosine_similarity'] for result in results if not result['paraphrase'] and not result['judgment']]
    incorrectly_judged_paraphrases =  [result['cosine_similarity'] for result in results if result['paraphrase'] and not result['judgment']]
    incorrectly_judged_non_paraphrases =  [result['cosine_similarity'] for result in results if not result['paraphrase'] and result['judgment']]

    return {
        'average_cosine_sim_for_correctly_judged_paraphrases': handle_zero_case(correctly_judged_paraphrases),
        'average_cosine_sim_for_correctly_judged_non_paraphrases': handle_zero_case(correctly_judged_non_paraphrases),
        'average_cosine_sim_for_incorrectly_judged_paraphrases': handle_zero_case(incorrectly_judged_paraphrases),
        'average_cosine_sim_for_incorrectly_judged_non_paraphrases': handle_zero_case(incorrectly_judged_non_paraphrases)
    }

def handle_zero_case(category_results):
    if not category_results:
        return 'N/A'
    return mean(category_results)


# PCA visualization code
# TODO: figure out what we actually want to include/disclude here and clean up

def PCA_comparisions(dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes):
    data = dataset.get_data()
    for idiom_sent_index in idiom_sentence_indexes:
        idiom_ex = data[idiom_sent_index]
        idiom_word_embedding = get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, idiom_sent_index)
        literal_usage_sents = [i for i, ex in enumerate(data) if ex.pair_id == idiom_ex.pair_id and 
                                                                ex.word == idiom_ex.word and not 
                                                                ex.sentence_id == idiom_ex.sentence_id ]
        paraphrase_sents = [i for i, ex in enumerate(data) if ex.pair_id == idiom_ex.pair_id 
                                                                and not ex.word == idiom_ex.word]

        literal_usage_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, lit_idx) for lit_idx in literal_usage_sents]
        paraphrase_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, para_idx) for para_idx in paraphrase_sents]

        title = 'PCA for {}: Fig usage {} Paraphrase word: {}'.format(idiom_ex.word[0], " ".join(idiom_ex.sentence), data[paraphrase_sents[0]].word)
        targets = {
            'labels': ['figurative', 'literal'],
            'values': [0, 1] ,
            'colors': ['turquoise', 'navy'],
        }
        labels =  np.array(len(literal_usage_embeddings) * [1] + [0])
        show_PCS(literal_usage_embeddings + [idiom_word_embedding], labels, targets, title)

        title = 'PCA for {}: Fig usage {}'.format(idiom_ex.word[0], " ".join(idiom_ex.sentence))
        targets = {
            'labels': ['figurative', 'literal', 'paraphrase'],
            'values': [0, 1, 2] ,
            'colors': ['turquoise', 'navy', 'orangered'],
        }
        embeddings = literal_usage_embeddings + [idiom_word_embedding] + paraphrase_embeddings
        labels =  np.array(len(literal_usage_embeddings) * [1] + [0] + len(paraphrase_embeddings) * [2])
        show_PCS(embeddings, labels, targets, title)

def show_PCS(embeddings, labels, targets, title):
    pca = PCA(2)  
    projected = pca.fit_transform(torch.stack(embeddings))

    for color, i, target_name in zip(targets['colors'], targets['values'], targets['labels']):
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1], color=color,  lw=2,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_batch_size', type=int, default=20)
    parser.add_argument('--embedding_cache', type=str, help='Directory to load cached embeddings from')
    parser.add_argument('--embedding_model', type=str, default='bert-large-uncased',
                        help='The model used to transform text into word embeddings')
    parser.add_argument('--embedding_paradigm', type=str, choices=[combined, cls], default=combined,
                        help='Whether to combine sentence embeddings or take the CLS token of joint embeddings')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='run_{}'.format((int(time.time()))),
                        help='A label for the run, used to name output and cache directories')
    parser.add_argument('--comparison_type', type=str, required=True)

    input_args = parser.parse_args()
    main(input_args)


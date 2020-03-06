import os.path
import sys
sys.path.append('../')
import torch
import itertools
import argparse
import time
import random
import logging
from statistics import mean 
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from probe.load_data import WordInspectionDataset, SentenceParaphraseInspectionDataset

words, paraphrase_sent_pairs = 'words', 'para_pairs'

def main(input_args):
    if input_args.comparison_type == words:
        word_usage_comparisons(input_args)
    else:
        sentence_paraphrase_comparisons(input_args)

def sentence_paraphrase_comparisons(input_args):
    dataset = SentenceParaphraseInspectionDataset(input_args.input, input_args.embedding_model, 
                                                input_args.embedding_batch_size, input_args.run_name)
    embeddings = get_embeddings(dataset, input_args.embedding_cache, flattened=True)
    embedding_outputs, encoded_inputs, _indices, _pools = embeddings
    sentence_embeddings = get_sentence_embeddings(embeddings, dataset)

    paraphrase_cosine_metrics = calculate_sent_cosine_metrics(dataset, embedding_outputs, 
                                                                            encoded_inputs, sentence_embeddings)
    # print(paraphrase_cosine_metrics)
    print(summarize_sentence_similarity_comp(paraphrase_cosine_metrics))

def word_usage_comparisons(input_args):
    dataset = WordInspectionDataset(input_args.input, input_args.embedding_model, 
                                    input_args.embedding_batch_size, input_args.run_name)
    embeddings = get_embeddings(dataset, input_args.embedding_cache, flattened=False)
    embedding_outputs, encoded_inputs, _indices, _pools = embeddings
    data = dataset.get_data()
    idiom_sentence_indexes = get_idiom_sentences(data)

    word_sim_results = calculate_word_cosine_metrics(dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes)
    avergages = summarize_word_similarity_comp(word_sim_results)
    
    individual_word_sims = list(itertools.chain.from_iterable([format_for_output(result) for result in word_sim_results]))
    embedding_meta_data_info_lines = run_information(input_args)
    output_lines = embedding_meta_data_info_lines + individual_word_sims + ["\nAverages:\n"] + format_for_output(avergages)
    output_file(input_args.run_name, '{}_word_similarity_results.tsv'.format(input_args.run_name), output_lines)

    PCA_comparisions(input_args.show_pca, input_args.run_name, dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes)

def run_information(input_args):
    return [
        'Run name: {} \n'.format(input_args.run_name),
        'Embedding model: {}\n'.format(input_args.embedding_model),
        'Embedding cache: {}\n'.format(input_args.embedding_cache),
        'Input file: {}\n\n\n'.format(input_args.input)
    ]

def format_for_output(metric_dict):
    return ["{}: {}\n".format(k, v) for k, v in metric_dict.items()] + ["\n"]

def get_embeddings(data, embedding_cache, flattened):
    if embedding_cache is None:
        if flattened:
            encoded_data = data.get_flattened_encoded()
        else:
            encoded_data = data.get_encoded()
        return data.bert_word_embeddings(encoded_data)
    return data.load_saved_embeddings(embedding_cache)

def get_sentence_embeddings(embeddings, data):
    embedding_outputs, encoded_inputs, indices, _pools = embeddings
    return data.aggregate_sentence_embeddings(embedding_outputs, encoded_inputs, indices)

def calculate_sent_cosine_metrics(dataset, embedding_outputs, encoded_inputs, sentence_embeddings):
    data = dataset.get_data()
    paraphrase_cosine_metrics = [calculate_paraphrase_pair_similarity(i, pair_sents, dataset, encoded_inputs, sentence_embeddings) 
                                for i, pair_sents in enumerate(data)]
    return paraphrase_cosine_metrics

def calculate_word_cosine_metrics(dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes):
    word_cosine_metrics = [calculate_word_similarity_metrics(idiom_sent_idx_group, dataset, 
                                                             embedding_outputs, encoded_inputs) 
                            for idiom_sent_idx_group in idiom_sentence_indexes]
    return word_cosine_metrics

def calculate_paraphrase_pair_similarity(index, classifier_out, dataset, encoded_inputs, sentence_embeddings):
    cosine_sim = 1 - cosine(sentence_embeddings[index][0], sentence_embeddings[index][1])
    return {
        'pair_index': index,
        'sent_1': " ".join(classifier_out.sentence_1),
        'sent_2': " ".join(classifier_out.sentence_2),
        'paraphrase': classifier_out.label,
        'judgment': classifier_out.classifier_judgment,
        'cosine_similarity': cosine_sim
    }    

def calculate_word_similarity_metrics(idiom_sent_index_group, dataset, embedding_outputs, encoded_inputs, random_indexes=None):
    data = dataset.get_data()
    idiom_exs = [data[idiom_sent_index] for idiom_sent_index in idiom_sent_index_group]
    idiom_word_embeddings = [get_word_embedding(dataset, data, embedding_outputs, 
                                               encoded_inputs, idiom_sent_index) 
                            for idiom_sent_index in idiom_sent_index_group]
    
    pair_id = idiom_exs[0].pair_id
    idiom_word = idiom_exs[0].word

    literal_usage_sents = [i for i, ex in enumerate(data) if ex.pair_id == pair_id and 
                                                            ex.word == idiom_word and not 
                                                            ex.figurative ]
    paraphrase_sents = [i for i, ex in enumerate(data) if ex.pair_id == pair_id 
                                                            and not ex.word == idiom_word]

    literal_usage_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, lit_idx) 
                                for lit_idx in literal_usage_sents]
    paraphrase_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, para_idx) 
                             for para_idx in paraphrase_sents]

    random_embeddings = None 
    if random_indexes:
        random_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, random_idx) 
                             for random_idx in random_indexes]

    return {
        'pair_id': pair_id,
        'idiom_sentences': [dataset.decode(encoded_inputs[idiom_sent_index].tolist()) for idiom_sent_index in idiom_sent_index_group],
        'word': idiom_word,
        'paraphrase_word': data[paraphrase_sents[0]].word,
        'cosine_similarities': calculate_word_cosine_sim_metrics(idiom_word_embeddings, literal_usage_embeddings, paraphrase_embeddings, random_embeddings),
        'euclidean_distances': calculate_word_euclidean_dists(idiom_word_embeddings, literal_usage_embeddings, paraphrase_embeddings, random_embeddings)
    }

def calculate_word_cosine_sim_metrics(idiom_word_embeddings, literal_usage_embeddings, paraphrase_embeddings, random_embeddings=None):
    cosine_similarity_metrics = {}
    cosine_similarity_metrics['fig_to_literal'] = calculate_dist_averages(cosine, True, idiom_word_embeddings, literal_usage_embeddings)
    cosine_similarity_metrics['literal_to_literal'] = calculate_dist_averages(cosine, True, literal_usage_embeddings)
    cosine_similarity_metrics['fig_to_fig'] = calculate_dist_averages(cosine, True, idiom_word_embeddings)
    cosine_similarity_metrics['fig_to_paraphrase'] = calculate_dist_averages(cosine, True, idiom_word_embeddings, paraphrase_embeddings)
    cosine_similarity_metrics['literal_to_paraphrase'] = calculate_dist_averages(cosine, True, literal_usage_embeddings, paraphrase_embeddings)

    if random_embeddings:
        cosine_similarity_metrics['fig_to_random'] = calculate_dist_averages(cosine, True, idiom_word_embeddings, random_embeddings)
    return cosine_similarity_metrics

def calculate_word_euclidean_dists(idiom_word_embeddings, literal_usage_embeddings, paraphrase_embeddings, random_embeddings=None):
    euclidean_dist_metrics = {}
    euclidean_dist_metrics['fig_to_literal'] = calculate_dist_averages(euclidean, False, idiom_word_embeddings, literal_usage_embeddings)
    euclidean_dist_metrics['literal_to_literal'] = calculate_dist_averages(euclidean, False, literal_usage_embeddings)
    euclidean_dist_metrics['fig_to_fig'] = calculate_dist_averages(euclidean, False, idiom_word_embeddings)
    euclidean_dist_metrics['fig_to_paraphrase'] = calculate_dist_averages(euclidean, False, idiom_word_embeddings, paraphrase_embeddings)
    euclidean_dist_metrics['literal_to_paraphrase'] = calculate_dist_averages(euclidean, False, literal_usage_embeddings, paraphrase_embeddings)

    if random_embeddings:
        euclidean_dist_metrics['fig_to_random'] = calculate_dist_averages(euclidean, False, idiom_word_embeddings, random_embeddings)
    return euclidean_dist_metrics
    
def get_idiom_sentences(dataset):
    idioms = [(i, ex) for i, ex in enumerate(dataset) if ex.figurative]
    values = set(map(lambda x:x[1].pair_id, idioms))
    return [[y[0] for y in idioms if y[1].pair_id == x] for x in values]

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

# for sanity checking the dataset embeddings and encoded outputs are aligned:
#   print(ex.sentence)
#   print(decoded_tokens)

    word_index = decoded_tokens.index(ex.word[0])
    return embedding_outputs[dataset_index][word_index]

def calculate_dist_averages(measurement, inverse, embeddings_1, embeddings_2=None):
    if embeddings_2:
        embedding_pairs = list(itertools.product(embeddings_1, embeddings_2))
    else:
        embedding_pairs = list(itertools.combinations(embeddings_1, 2))

    if inverse:
        distances = [1 - measurement(embedding_1, embedding_2) for embedding_1, embedding_2 in embedding_pairs]
    else:
        distances = [measurement(embedding_1, embedding_2) for embedding_1, embedding_2 in embedding_pairs]
    return mean(distances)


# This computes the average difference in cosine similarity between:
# 1.) literal to literal usages versus figurative to literal usage
# 2.) figurative to paraphrase usages versus literal to paraphrase useage
def summarize_word_similarity_comp(results):
    cosine_literal_sim_advantage = [result['cosine_similarities']['literal_to_literal'] - result['cosine_similarities']['fig_to_literal'] for result in results]
    cosine_fig_to_paraphrase_advantage = [result['cosine_similarities']['fig_to_paraphrase'] - result['cosine_similarities']['literal_to_paraphrase'] for result in results]

    eud_literal_sim_advantage = [result['euclidean_distances']['fig_to_literal'] - result['euclidean_distances']['literal_to_literal']  for result in results]
    eud_fig_to_paraphrase_advantage = [result['euclidean_distances']['literal_to_paraphrase'] - result['euclidean_distances']['fig_to_paraphrase'] for result in results]

    summary_stats = {
        'COSINE SIM- lit_to_lit_improvement_over_fig_to_lit': mean(cosine_literal_sim_advantage),
        'COSINE SIM- fig_to_paraphrase_improvement_over_lit_to_paraphrase': mean(cosine_fig_to_paraphrase_advantage),
        'EUCLIDEAN DIST- lit_to_lit_improvement_over_fig_to_lit': mean(eud_literal_sim_advantage),
        'EUCLIDEAN DIST- fig_to_paraphrase_improvement_over_lit_to_paraphrase': mean(eud_fig_to_paraphrase_advantage)

    }
    return summary_stats


# This computes the average cosine similarity scores between paraphrase pairs,
# grouped into 4 categories based on gold label (i.e. true paraphrase or not) and classifier judgment
def summarize_sentence_similarity_comp(results):
    correctly_judged_paraphrases = [result['cosine_similarity'] for result in results if result['paraphrase'] and result['judgment']]
    correctly_judged_non_paraphrases = [result['cosine_similarity'] for result in results if not result['paraphrase'] and not result['judgment']]
    incorrectly_judged_paraphrases =  [result['cosine_similarity'] for result in results if result['paraphrase'] and not result['judgment']]
    incorrectly_judged_non_paraphrases =  [result['cosine_similarity'] for result in results if not result['paraphrase'] and result['judgment']]

    paraphrases = [result['cosine_similarity'] for result in results if result['paraphrase']]
    non_paraphrases = [result['cosine_similarity'] for result in results if not result['paraphrase']]

    return {
        'average_cosine_sim_for_correctly_judged_paraphrases': handle_zero_case(correctly_judged_paraphrases),
        'average_cosine_sim_for_correctly_judged_non_paraphrases': handle_zero_case(correctly_judged_non_paraphrases),
        'average_cosine_sim_for_incorrectly_judged_paraphrases': handle_zero_case(incorrectly_judged_paraphrases),
        'average_cosine_sim_for_incorrectly_judged_non_paraphrases': handle_zero_case(incorrectly_judged_non_paraphrases),
        'average_cosine_for_paraphrases': handle_zero_case(paraphrases),
        'average_cosine_for_non_paraphrases': handle_zero_case(non_paraphrases)
    }

def handle_zero_case(category_results):
    if not category_results:
        return 'N/A'
    return mean(category_results)


# PCA visualization code
def PCA_comparisions(show_image, run_name, dataset, embedding_outputs, encoded_inputs, idiom_sentence_indexes):
    data = dataset.get_data()
    for num, idiom_sent_index_group in enumerate(idiom_sentence_indexes):
        idiom_exs = [data[idiom_sent_index] for idiom_sent_index in idiom_sent_index_group]
        idiom_word_embeddings = [get_word_embedding(dataset, data, embedding_outputs, 
                                               encoded_inputs, idiom_sent_index) 
                                for idiom_sent_index in idiom_sent_index_group]
        pair_id = idiom_exs[0].pair_id
        idiom_word = idiom_exs[0].word[0]
        literal_usage_sents = [i for i, ex in enumerate(data) if ex.pair_id == pair_id and 
                                                                ex.word[0] == idiom_word and not ex.figurative ]
        paraphrase_sents = [i for i, ex in enumerate(data) if ex.pair_id == pair_id and not ex.word[0] == idiom_word]

        literal_usage_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, lit_idx) 
                                    for lit_idx in literal_usage_sents]
        
        paraphrase_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, para_idx) 
                                 for para_idx in paraphrase_sents]

        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

        literal_labels = len(literal_usage_embeddings) * [1]
        idiom_labels = len(idiom_word_embeddings) * [0]
        paraphrase_labels = len(paraphrase_embeddings) * [2]
        
        print("\n\n\n\n\nExample " + str(num))
        print("Figurative phrases:  ")
        for idiom_ex in idiom_exs:
            print(" ".join(idiom_ex.sentence))
        print()
        print("Idiom usage word:  " + idiom_word)
        print("Paraphrase word:  " + data[paraphrase_sents[0]].word[0])
        
        # First PCA graph: only literal and figurative usages
        title = 'PCA for: "{}"'.format(idiom_word)
        targets = {
            'labels': ['figurative', 'literal'],
            'values': [0, 1] ,
            'colors': ['turquoise', 'navy'],
        }

        labels =  np.array(literal_labels + idiom_labels)
        image_filename = "pair_id_{}_fig_lit".format(pair_id)
        generate_PCS(run_name, literal_usage_embeddings + idiom_word_embeddings, labels, targets, title, image_filename, show_image)

        # Second PCA graph: literal, figurative, and paraphrases
        title = 'PCA for: "{}"; Paraphrase word: {}'.format(idiom_word, data[paraphrase_sents[0]].word)
        targets = {
            'labels': ['figurative', 'literal', 'paraphrase'],
            'values': [0, 1, 2] ,
            'colors': ['turquoise', 'navy', 'orangered'],
        }
        embeddings = literal_usage_embeddings + idiom_word_embeddings + paraphrase_embeddings
        labels =  np.array(literal_labels + idiom_labels + paraphrase_labels)
        image_filename = "pair_id_{}_fig_lit_para".format(pair_id)
        generate_PCS(run_name, embeddings, labels, targets, title, image_filename, show_image)

        # Third PCA graph: literal, figurative, paraphrases, and random word
        random_id = random.choice([999, 899, 799])
        random_word_sents = [i for i, ex in enumerate(data) if ex.pair_id == random_id]
        random_word_embeddings = [get_word_embedding(dataset, data, embedding_outputs, encoded_inputs, rnd_idx) 
                                  for rnd_idx in random_word_sents]
  
        title = 'PCA for: "{}"; Paraphrase word: {}; Random word: {}'.format(idiom_word, 
                                                                        data[paraphrase_sents[0]].word,
                                                                        data[random_word_sents[0]].word)
        targets = {
            'labels': ['figurative', 'literal', 'paraphrase', 'random'],
            'values': [0, 1, 2, 3] ,
            'colors': ['turquoise', 'navy', 'orangered', 'gray'],
        }
        
        embeddings = literal_usage_embeddings + idiom_word_embeddings + paraphrase_embeddings + random_word_embeddings
        labels =  np.array(literal_labels + idiom_labels + paraphrase_labels + len(random_word_embeddings) * [3])
        image_filename = "pair_id_{}_fig_lit_para_rand".format(pair_id)
        generate_PCS(run_name, embeddings, labels, targets, title, image_filename, show_image)
        
        word_sim_calculations = calculate_word_similarity_metrics(idiom_sent_index_group, dataset,
                                                                embedding_outputs, 
                                                                encoded_inputs, random_word_sents)
        word_cosine_results = word_sim_calculations['cosine_similarities']        
        word_euclidean_results = word_sim_calculations['euclidean_distances']        
        
        print("Cosine similarity scores (higher is 'closer')")
        for pair_type, cosine_val in word_cosine_results.items():
            print(pair_type, ": " + str(cosine_val))

        print("\n\nEuclidean distance scores (higher is 'further')")
        for pair_type, eucl_dist in word_euclidean_results.items():
            print(pair_type, ": " + str(eucl_dist))

def generate_PCS(run_name, embeddings, labels, targets, title, save_filename, show=False):
    create_PCS_output_folder(run_name)
    pca = PCA(2)  
    projected = pca.fit_transform(torch.stack(embeddings))

    for color, i, target_name in zip(targets['colors'], targets['values'], targets['labels']):
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1], color=color,  lw=2,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)

    if show:
        plt.show()
    else:
        plt.savefig('output/{}/PCA_images/{}'.format(run_name, save_filename))
        plt.clf()

def create_PCS_output_folder(run_name):
    folder = os.path.join('output', run_name, 'PCA_images')
    if not os.path.exists(folder):
        os.makedirs(folder)

def output_file(run_name, filename, content):
    folder = os.path.join('output', run_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), 'w+') as outfile:
        outfile.writelines(content)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_batch_size', type=int, default=20)
    parser.add_argument('--embedding_cache', type=str, help='Directory to load cached embeddings from')
    parser.add_argument('--embedding_model', type=str, default='bert-large-uncased',
                        help='The model used to transform text into word embeddings')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='run_{}'.format((int(time.time()))),
                        help='A label for the run, used to name output and cache directories')
    parser.add_argument('--comparison_type', type=str, required=True)
    parser.add_argument('--show_pca', type=str, default=False)

    input_args = parser.parse_args()
    main(input_args)


import sys
sys.path.append('../')
from transformers import BertTokenizer, BertModel
import torch
import itertools
from probe.load_data import WordInspectionDataset
from scipy.spatial.distance import cosine
from statistics import mean 

def main():
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    feature_extraction_model = BertModel.from_pretrained('bert-large-uncased')
    batch_size = 20  # totally arbitrarily chosen

    pdata = WordInspectionDataset('vec_sim_test.txt', tokenizer)
    dataset = pdata.get_data()
    embedding_outputs, encoded_inputs, indices = pdata.bert_word_embeddings(feature_extraction_model,
                                                                            pdata.get_encoded(), batch_size)
    sentence_embeddings = pdata.aggregate_sentence_embeddings(embedding_outputs, encoded_inputs, indices,
                                                              aggregation_metric=torch.mean)

    idiom_sentence_indexes = get_idiom_sentences(dataset)
    word_cosine_metrics = [calculate_similarity_metrics(idiom_sent_idx, tokenizer, dataset, embedding_outputs, encoded_inputs) 
                            for idiom_sent_idx in idiom_sentence_indexes]


def calculate_similarity_metrics(idiom_sent_index, tokenizer, dataset, embedding_outputs, encoded_inputs):
    idiom_ex = dataset[idiom_sent_index]
    idiom_word_embedding = get_word_embedding(tokenizer, dataset, embedding_outputs, encoded_inputs, idiom_sent_index)
    cosine_similarity_metrics = {}

    literal_usage_sents = [i for i, ex in enumerate(dataset) if ex.pair_id == idiom_ex.pair_id and 
                                                            ex.word == idiom_ex.word and not 
                                                            ex.sentence_id == idiom_ex.sentence_id ]
    paraphrase_sents = [i for i, ex in enumerate(dataset) if ex.pair_id == idiom_ex.pair_id 
                                                            and not ex.word == idiom_ex.word]

    literal_usage_embeddings = [get_word_embedding(tokenizer, dataset, embedding_outputs, encoded_inputs, lit_idx) for lit_idx in literal_usage_sents]
    paraphrase_embeddings = [get_word_embedding(tokenizer, dataset, embedding_outputs, encoded_inputs, para_idx) for para_idx in paraphrase_sents]

    cosine_similarity_metrics['fig_to_literal'] = calculate_cosine_similarity_average([idiom_word_embedding], literal_usage_embeddings)
    cosine_similarity_metrics['literal_to_literal'] = calculate_cosine_similarity_average(literal_usage_embeddings)
    cosine_similarity_metrics['fig_to_paraphrase'] = calculate_cosine_similarity_average([idiom_word_embedding], paraphrase_embeddings)
    cosine_similarity_metrics['literal_to_paraphrase'] = calculate_cosine_similarity_average(literal_usage_embeddings, paraphrase_embeddings)
    
    return {
        'sentence_id': idiom_ex.sentence_id,
        'sentence': idiom_ex.sentence,
        'word': idiom_ex.word,
        'paraphrase_word': dataset[paraphrase_sents[0]].word,
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

def get_word_embedding(tokenizer, dataset, embedding_outputs, encoded_inputs, dataset_index):
    ex = dataset[dataset_index]
    decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_inputs[dataset_index].tolist())
    word_index = decoded_tokens.index(ex.word[0])
    return embedding_outputs[dataset_index][word_index]

def calculate_cosine_similarity_average(embeddings_1, embeddings_2=None):
    if embeddings_2:
        embedding_pairs = list(itertools.product(embeddings_1, embeddings_2))
    else:
        embedding_pairs = list(itertools.combinations(embeddings_1, 2))

    cosine_similarities = [1 - cosine(embedding_1, embedding_2) for embedding_1, embedding_2 in embedding_pairs]
    return mean(cosine_similarities)


#TODO: calculate cosine similarity on a sentence level

#TODO: write averaging methods for generalizing about the overall cosine similarity relations

if __name__ =='__main__':
    main()


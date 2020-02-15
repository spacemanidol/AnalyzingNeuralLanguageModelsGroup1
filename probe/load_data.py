import torch
import numpy as np
import codecs
from torch.utils.data import ConcatDataset
from torchtext import data
from tqdm.autonotebook import tqdm

#TODO: download the data if it's not here
MSRP_URLS = ['https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_train.txt',
             'https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_test.txt']


def load_paraphrase_data(filename, tokenizer, indices=(0, 1, 2), csv_reader_params=None):
    if csv_reader_params is None:
        csv_reader_params = {'strict': True}

    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    sentence_field = data.Field(use_vocab=False,
            tokenize=lambda x: tokenizer.encode(x, add_special_tokens=True), pad_token=pad_index)
    label_field = data.LabelField(preprocessing=lambda x: int(x), use_vocab=False)

    field_array = [('unused', None)] * (max(indices) + 1)
    field_array[indices[0]] = ("label", label_field)
    field_array[indices[1]] = ("sentence_1", sentence_field)
    field_array[indices[2]] = ("sentence_2", sentence_field)

    paraphrases = data.TabularDataset(
        path=filename,
        format="tsv",  skip_header=True,
        fields=field_array,
        csv_reader_params=csv_reader_params
    )

    return paraphrases, sentence_field

# take X examples of sentence pairs and conver them into 2X rows of single sentences with pair IDS so they can be
# processed separately
def flatten_data_for_feature_extraction(paraphrase_data, sentence_field):
    fields = [('sentence', sentence_field), ('pair_id', data.LabelField(use_vocab=False))]
    return data.Dataset(
        [data.Example.fromlist([row.sentence_1, [index, 0]], fields) for index, row in enumerate(paraphrase_data)] +
        [data.Example.fromlist([row.sentence_2, [index, 1]], fields) for index, row in enumerate(paraphrase_data)],
        fields)

# take a tensor of shape [sentence count] x [sentence length] x [embedding length] and return a tensor of shape
# [sentence count] x [embedding length] by aggregating all included word embeddings into a single sentence embedding
# inclusion_matrix allows us to specify embeddings that we don't want aggregated (such as those that are generated
# from padding tokens). for Bert, we expect this to be the attention mask
def aggregate_sentence_embeddings(embeddings_matrix, inclusion_matrix=None, aggregation_metric=torch.mean):
    if inclusion_matrix is None:
        inclusion_matrix = torch.ones(embeddings_matrix.shape[0], embeddings_matrix.shape[1])

    selection_matrix = inclusion_matrix.type(torch.BoolTensor)
    return torch.stack([
        aggregation_metric(sentence_row[selection_matrix[row_index]], axis=0)
        for row_index, sentence_row in enumerate(embeddings_matrix)
    ])

# unshuffle the sentence pairs and return a tensor of shape [pair counnt] x [2] x [embedding size]
def recover_sentence_pairs(sentence_embeddings, pair_ids):
    pairs = torch.zeros([sentence_embeddings.shape[0] // 2, 2, sentence_embeddings.shape[1]])
    for index, pair_id in enumerate(pair_ids):
        pairs[tuple(pair_id)] = sentence_embeddings[index]
    return pairs

# takes pairs of sentences from recover_sentence_pairs and combines them
def combine_sentence_embeddings(sentence_embedding_pairs, combination_metric=torch.sub):
    return torch.stack([combination_metric(pair[0], pair[1]) for pair in sentence_embedding_pairs])


def get_msrp_combined_embeddings(tokenizer, feature_extraction_model):
    batch_size = 20
    paraphrases_data, sentence_field = load_paraphrase_data('test_head.txt', tokenizer, indices=(0, 3, 4))
    # labels must be floats (not ints) or the function to compute loss gags
    labels = torch.tensor([x.label for x in paraphrases_data], dtype=torch.float32)
    flattened_data = flatten_data_for_feature_extraction(paraphrases_data, sentence_field)

    # BucketIterator transposes the sentence data, we have to transpose it back
    # pair ID data does not need to be transposed (perhaps because it is one dimensional?)
    batch_ids_iterator = (({'input_ids': tensor,
                           'attention_mask': (tensor != 0) * 1}, pair_ids)
                        for tensor, pair_ids in ((x.sentence.transpose(0, 1), x.pair_id)
                        for x in data.BucketIterator(flattened_data, batch_size, sort_key=lambda x: len(x.sentence))))

    embeddings = None
    pair_ids = None
    attention_masks = None
    for batch, ids in tqdm(batch_ids_iterator, desc="Feature extraction"):
        with torch.no_grad():
            out = feature_extraction_model(**batch)[0]
            mask = batch['attention_mask']
            embeddings = out if not embeddings else torch.cat((embeddings, out))
            pair_ids = ids if not pair_ids else torch.cat((pair_ids, ids))
            attention_masks = mask if not attention_masks else torch.cat((attention_masks, mask))

    sentence_embeddings = aggregate_sentence_embeddings(embeddings, attention_masks)
    pairs = recover_sentence_pairs(sentence_embeddings, pair_ids)
    paraphrase_embeddings = combine_sentence_embeddings(pairs)

    return paraphrase_embeddings, labels




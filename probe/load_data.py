import torch
from functools import reduce
from torchtext import data
from tqdm.autonotebook import tqdm
from abc import ABC, abstractmethod

#TODO: download the data if it's not here
MSRP_URLS = ['https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_train.txt',
             'https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_test.txt']


class Dataset(ABC):
    def get_data(self):
        if self.data is None:
            self.load()

        return self.data

    def __init__(self, filename, tokenizer):
        self.filename = filename
        self.tokenizer = tokenizer
        self.data = None
        self.encoded_sentence_field = ('sentence',
                                       data.Field(use_vocab=False, tokenize=self.encode, pad_token=self.tokenizer.pad_token_id))
        self.index_field = ('index', data.LabelField(use_vocab=False))
        self.encoded_fields = [self.encoded_sentence_field, self.index_field]


    def encode(self, sentence):
        return self.tokenizer.encode(sentence, add_special_tokens=True)

    @abstractmethod
    def load(self):
        raise Exception("Not implemented on for this class")

    def bert_iter(self, encoded_data, batch_size):
        # BucketIterator transposes the sentence data, we have to transpose it back
        # pair ID data does not need to be transposed (perhaps because it is one dimensional?)
        return (({'input_ids': tensor,
                  'attention_mask': (tensor != self.tokenizer.pad_token_id) * 1}, indices)
                for tensor, indices in ((x.sentence.transpose(0, 1), x.index)
                for x in data.BucketIterator(encoded_data, batch_size, sort_key=lambda x: len(x.sentence))))

    def bert_word_embeddings(self, bert_model, encoded_data, batch_size):
        sentences = None
        indices = None
        inputs = None
        for batch, ids in tqdm(self.bert_iter(encoded_data, batch_size), desc="Feature extraction"):
            with torch.no_grad():
                out = bert_model(**batch)[0]
                mask = batch['input_ids']
                sentences = out if not sentences else torch.cat((sentences, out))
                indices = ids if not indices else torch.cat((indices, ids))
                inputs = mask if not inputs else torch.cat((inputs, mask))

        return self.reorder(sentences, inputs, indices)

    @staticmethod
    def reorder(sentences, inputs, indices):
        if len(indices.shape) > 1:
            grouping_length = indices.shape[1]
            # dealing with a grouping of sentences, like pairs
            ordered_sentences = torch.zeros([sentences.shape[0] // grouping_length, grouping_length,
                                             sentences.shape[1], sentences.shape[2]])
            ordered_inputs = torch.zeros(inputs.shape[0] // grouping_length, grouping_length, inputs.shape[1])
            indices = [tuple(x) for x in indices]
            ordered_indices = sorted(indices)
        else:
            ordered_sentences = torch.zeros(sentences.shape)
            ordered_inputs = torch.zeros(inputs.shape)
            ordered_indices = range(0, sentences.shape[0])

        for current_index, original_index in enumerate(indices):
            ordered_sentences[original_index] = sentences[current_index]
            ordered_inputs[original_index] = inputs[current_index]

        return ordered_sentences, ordered_inputs, ordered_indices


    def aggregate_sentence_embeddings(self, ordered_sentences, ordered_inputs, ordered_indices,
                                      aggregation_metric=torch.mean):
        selection_matrix = reduce(lambda x,y: x & y, (ordered_inputs != x for x in self.tokenizer.all_special_ids))

        output_dimensions = list(ordered_sentences.shape)
        del output_dimensions[-2] # remove the second last to dimensions, which is the token count
        output = torch.zeros(torch.Size(output_dimensions))

        for index in ordered_indices:
            output[index] = aggregation_metric(ordered_sentences[index][selection_matrix[index]], axis=0)

        return output



# for MSR paraphrase data and our paraphrase data
class ParaphraseDataset(Dataset):
    def __init__(self, filename, tokenizer, indices=(0, 1, 2)):
        super().__init__(filename, tokenizer)
        self.flattened_encoded_data = None
        self.encoded_data = None
        self.labels = None
        self.indices = indices

    def get_flattened_encoded(self):
        if self.flattened_encoded_data is None:
            self.__compute_flattened_encoded()

        return self.flattened_encoded_data

    def get_labels(self):
        if self.labels is None:
            # labels must be floats (not ints) or the function to compute loss gags
            self.labels = torch.tensor([x.label for x in self.get_data()], dtype=torch.float32)

        return self.labels

    def load(self):
        indices = self.indices
        tokenized_field = data.Field(use_vocab=False, tokenize=lambda x: self.tokenizer.tokenize(x))
        label_field = data.LabelField(preprocessing=lambda x: int(x), use_vocab=False)

        field_array = [('unused', None)] * (max(indices) + 1)
        field_array[indices[0]] = ("label", label_field)
        field_array[indices[1]] = ("sentence_1", tokenized_field)
        field_array[indices[2]] = ("sentence_2", tokenized_field)

        self.data = data.TabularDataset(
            path=self.filename,
            format="tsv",  skip_header=True,
            fields=field_array,
            csv_reader_params={'strict': True}
        )

    # take X examples of sentence pairs and convert them into 2X rows of encoded single sentences with pair IDS so they
    # can be processed separately by bert
    def __compute_flattened_encoded(self):
        paraphrase_data = self.get_data()
        self.flattened_encoded_data =  data.Dataset(
            [data.Example.fromlist([self.encode(row.sentence_1), [index, 0]], self.encoded_fields) for index, row in
             enumerate(paraphrase_data)] +
            [data.Example.fromlist([self.encode(row.sentence_2), [index, 1]], self.encoded_fields) for index, row in
             enumerate(paraphrase_data)],
            self.encoded_fields)

    @staticmethod
    def combine_sentence_embeddings(sentence_embedding_pairs, combination_metric=torch.sub):
        return torch.stack([combination_metric(pair[0], pair[1]) for pair in sentence_embedding_pairs])


# for Paige's word vector similarity
class WordInspectionDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super().__init__(filename, tokenizer)
        self.data = None
        self.encoded_data = None

    def get_data(self):
        if self.data is None:
            self.load()

        return self.data

    def get_encoded(self):
        if self.encoded_data is None:
            self.__compute_encoded()

        return self.encoded_data

    def load(self):
        tokenized_field = data.Field(use_vocab=False, tokenize=lambda x: self.tokenizer.tokenize(x))
        label_field = data.LabelField(preprocessing=lambda x: int(x), use_vocab=False)
        fields = [
            ('sentence_id', label_field),
            ('pair_id', label_field),
            ('sentence', tokenized_field),
            ('word', tokenized_field)
        ]

        self.data = data.TabularDataset(
            path=self.filename,
            format="tsv",  skip_header=True,
            fields=fields,
            csv_reader_params={'strict': True}
        )

    def __compute_encoded(self):
        self.encoded_data = data.Dataset(
            [data.Example.fromlist([self.encode(row.sentence), index], self.encoded_fields) for index, row in
             enumerate(self.get_data())],
            self.encoded_fields)


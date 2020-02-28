import torch
from functools import reduce
from torchtext import data
from tqdm.autonotebook import tqdm
from abc import ABC, abstractmethod
import logging
import os.path
import json
import csv
from transformers import AutoTokenizer, BertModel

logging.basicConfig(level=logging.DEBUG)
module_logger = logging.getLogger('data_loading')

class Dataset(ABC):

    sentences_filename = 'sentences.pt'
    inputs_filename = 'inputs.pt'
    indices_filename = 'indices.pt'

    def get_raw(self):
        with open(self.filename, 'r') as f:
            out = [x for x in csv.reader(f, delimiter='\t', quotechar=None, strict=True)]

        return out


    def get_data(self):
        if self.data is None:
            self.load()

        return self.data

    def get_encoded(self):
        if self.encoded_data is None:
            self._compute_encoded()

        return self.encoded_data

    def __init__(self, filename, model_label, batch_size, run_name):
        self.run_name = run_name
        self.filename = filename
        self.model_label = model_label
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_label, do_lower_case=True)
        self.data = None
        self.encoded_sentence_field = ('sentence',
                                       data.Field(use_vocab=False, tokenize=self.encode, pad_token=self.tokenizer.pad_token_id))
        self.index_field = ('index', data.LabelField(use_vocab=False))
        self.encoded_fields = [self.encoded_sentence_field, self.index_field]

    def encode(self, sentence, second_sentence=None):
        return self.tokenizer.encode(sentence, second_sentence, add_special_tokens=True)

    def decode(self, sentence):
        return self.tokenizer.decode(sentence)

    def get_decoded_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    @abstractmethod
    def load(self):
        raise Exception("Not implemented on for this class")

    @abstractmethod
    def _compute_encoded(self):
        raise Exception("Not implemented on for this class")


    def bert_iter(self, encoded_data, batch_size):
        # BucketIterator transposes the sentence data, we have to transpose it back
        # pair ID data does not need to be transposed (perhaps because it is one dimensional?)
        return (({'input_ids': tensor,
                  'attention_mask': (tensor != self.tokenizer.pad_token_id) * 1}, indices)
                for tensor, indices in ((x.sentence.transpose(0, 1), x.index)
                for x in data.BucketIterator(encoded_data, batch_size, sort_key=lambda x: len(x.sentence))))


    @staticmethod
    def aggregate_data(sentences, batch_sentences, indices, batch_indices, inputs, batch_inputs):
        # padding solution is pretty hacky and probably not the most space efficient, but it works for what
        # we want to do
        if sentences is None and indices is None and inputs is None:
            sentences = batch_sentences
            indices = batch_indices
            inputs = batch_inputs
        else:
            indices = torch.cat((indices, batch_indices))
            if sentences.shape[1] > batch_sentences.shape[1]:
                #existing tensor is bigger, pad inputs
                batch_sentences = torch.nn.functional.pad(batch_sentences,
                                                          (0, 0, 0, sentences.shape[1] - batch_sentences.shape[1]))
                batch_inputs = torch.nn.functional.pad(batch_inputs, (0, inputs.shape[1] - batch_inputs.shape[1]))
            elif sentences.shape[1] < batch_sentences.shape[1]:
                #inputs are bigger, pad existing
                sentences = torch.nn.functional.pad(sentences,
                                                          (0, 0, 0, batch_sentences.shape[1] - sentences.shape[1]))
                inputs = torch.nn.functional.pad(inputs, (0, batch_inputs.shape[1] - inputs.shape[1]))

            sentences = torch.cat((sentences, batch_sentences))
            inputs = torch.cat((inputs, batch_inputs))

        return sentences, indices, inputs

    def _save(self, tensor, name, folder):
        module_logger.info('Caching {} in {}'.format(name, folder))
        torch.save(tensor, os.path.join(folder, name))

    def _load(self, name, folder):
        module_logger.info('Loading {} from {}'.format(name, folder))
        return torch.load(os.path.join(folder, name))

    def get_metadata(self):
        return {
            'file': self.filename,
            'model': self.model_label,
            'batch_size': self.batch_size,
            'run_name': self.run_name
        }

    def save_computed_embeddings(self, sentences, inputs, indices, metadata):
        folder = os.path.join('cache', self.run_name)
        module_logger.info('Caching info for this run in {}'.format(folder))
        module_logger.info('Please pass this folder in to future invocations to use cached data')
        if not os.path.exists(folder):
            os.makedirs(folder)

        if metadata is not None:
            with open(os.path.join(folder, 'metadata.json'), 'w+') as metadata_file:
                metadata_file.write(json.dumps(metadata)+'\n')

        self._save(sentences, self.sentences_filename, folder)
        self._save(inputs, self.inputs_filename, folder)
        self._save(indices, self.indices_filename, folder)

    def load_saved_embeddings(self, folder):
        module_logger.info('Loading embedding data from {}...'.format(folder))
        return self._load(self.sentences_filename, folder), self._load(self.inputs_filename, folder),\
               self._load(self.indices_filename, folder),

    def bert_word_embeddings(self, encoded_data):
        module_logger.info("Loading '{}' model".format(self.model_label))
        bert_model = BertModel.from_pretrained(self.model_label)

        sentences = None
        indices = None
        inputs = None
        for batch, batch_indices in tqdm(self.bert_iter(encoded_data, self.batch_size), desc="Feature extraction"):
            with torch.no_grad():
                out = bert_model(**batch)[0]
                batch_inputs = batch['input_ids']
                sentences, indices, inputs = self.aggregate_data(sentences, out, indices, batch_indices,
                                                                 inputs, batch_inputs)

            module_logger.info('processed {}/{} sentences, current max sentence length {}'
                  .format(sentences.shape[0], len(encoded_data), sentences.shape[1]))

        ordered_sentences, ordered_inputs, ordered_indices = self.reorder(sentences, inputs, indices)
        self.save_computed_embeddings(ordered_sentences, ordered_inputs, ordered_indices, self.get_metadata())
        return ordered_sentences, ordered_inputs, ordered_indices

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
    def __init__(self, filename, model_label, batch_size, run_name, indices=(0, 1, 2)):
        super().__init__(filename, model_label, batch_size, run_name)
        self.flattened_encoded_data = None
        self.encoded_data = None
        self.labels = None
        self.indices = indices

    def get_raw_for_output(self):
        indices = self.indices
        raw_data = self.get_raw()
        row_1 = ('true_label', 'sentence_1', 'sentence_2') + tuple(x for index, x in enumerate(raw_data[0])
                                                              if index not in indices)
        return [row_1] + [
            (row[indices[0]], row[indices[1]], row[indices[2]]) +
            tuple(x for index, x in enumerate(row) if index not in indices) for row in raw_data[1:]
        ]

    def get_flattened_encoded(self):
        if self.flattened_encoded_data is None:
            self._compute_flattened_encoded()

        return self.flattened_encoded_data

    def get_labels(self):
        if self.labels is None:
            # labels must be floats (not ints) or the function to compute loss gags
            self.labels = torch.tensor([x.label for x in self.get_data()], dtype=torch.float32).unsqueeze(1)

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
            csv_reader_params={'strict': True, 'quotechar':None}
        )

    # take X examples of sentence pairs and convert them into 2X rows of encoded single sentences with pair IDS so they
    # can be processed separately by bert
    def _compute_flattened_encoded(self):
        paraphrase_data = self.get_data()
        self.flattened_encoded_data = data.Dataset(
            [data.Example.fromlist([self.encode(row.sentence_1), [index, 0]], self.encoded_fields) for index, row in
             enumerate(paraphrase_data)] +
            [data.Example.fromlist([self.encode(row.sentence_2), [index, 1]], self.encoded_fields) for index, row in
             enumerate(paraphrase_data)],
            self.encoded_fields)

    def _compute_encoded(self):
        paraphrase_data = self.get_data()
        self.encoded_data = data.Dataset(
            [data.Example.fromlist([self.encode(row.sentence_1, row.sentence_2), index], self.encoded_fields)
             for index, row in enumerate(paraphrase_data)],
            self.encoded_fields)


    @staticmethod
    def combine_sentence_embeddings(sentence_embedding_pairs, combination_metric=torch.sub):
        return torch.stack([combination_metric(pair[0], pair[1]) for pair in sentence_embedding_pairs])

    @staticmethod
    def bert_cls_embeddings(sentence_embeddings):
        return sentence_embeddings[:,0]



# for Paige's word vector similarity
class WordInspectionDataset(Dataset):
    def __init__(self, filename, model_label, batch_size, run_name):
        super().__init__(filename, model_label, batch_size, run_name)
        self.data = None
        self.encoded_data = None


    def load(self):
        tokenized_field = data.Field(use_vocab=False, tokenize=lambda x: self.tokenizer.tokenize(x))
        label_field = data.LabelField(preprocessing=lambda x: int(x), use_vocab=False)
        fields = [
            ('sentence_id', label_field),
            ('pair_id', label_field),
            ('sentence', tokenized_field),
            ('word', tokenized_field),
            ('figurative', label_field)
        ]

        self.data = data.TabularDataset(
            path=self.filename,
            format="tsv",  skip_header=True,
            fields=fields,
            csv_reader_params={'strict': True, 'quotechar': None}
        )

    def _compute_encoded(self):
        self.encoded_data = data.Dataset(
            [data.Example.fromlist([self.encode(row.sentence), index], self.encoded_fields) for index, row in
             enumerate(self.get_data())],
            self.encoded_fields)


class SentenceParaphraseInspectionDataset(WordInspectionDataset):
    def load(self):
        tokenized_field = data.Field(use_vocab=False, tokenize=lambda x: self.tokenizer.tokenize(x))
        label_field = data.LabelField(preprocessing=lambda x: int(x), use_vocab=False)
        fields = [
            ('sentence_id', label_field),
            ('pair_id', label_field),
            ('sentence', tokenized_field),
            ('paraphrase', label_field),
            ('classifier_judgment', label_field)
        ]

        self.data = data.TabularDataset(
            path=self.filename,
            format="tsv",  skip_header=True,
            fields=fields,
            csv_reader_params={'strict': True, 'quotechar': None}
        )


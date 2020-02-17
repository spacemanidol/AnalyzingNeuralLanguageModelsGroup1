from transformers import BertTokenizer, BertModel
import torch
from load_data import WordInspectionDataset


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    feature_extraction_model = BertModel.from_pretrained('bert-large-uncased')
    batch_size = 20  # totally arbitrarily chosen

    pdata = WordInspectionDataset('paige_example.txt', tokenizer)
    dataset = pdata.get_data()
    embedding_outputs, encoded_inputs, indices = pdata.bert_word_embeddings(feature_extraction_model,
                                                                            pdata.get_encoded(), batch_size)
    sentence_embeddings = pdata.aggregate_sentence_embeddings(embedding_outputs, encoded_inputs, indices,
                                                              aggregation_metric=torch.mean)


if __name__ =='__main__':
    main()

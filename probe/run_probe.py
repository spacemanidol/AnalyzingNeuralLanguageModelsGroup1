from transformers import BertTokenizer, BertModel
from torch.autograd import Variable
import torch
from load_data import ParaphraseDataset
import argparse
import logging
import time
import os.path


module_logger = logging.getLogger('probe')

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out



def train_probe(args):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    feature_extraction_model = BertModel.from_pretrained('bert-large-uncased')

    batch_size = 128
    #msr_train = ParaphraseDataset('msr_paraphrase_train.txt', tokenizer, indices=(0, 3, 4))
    msr_train = ParaphraseDataset('train_head_200.txt', tokenizer, indices=(0, 3, 4))
    labels = msr_train.get_labels()
    pairs, inputs, indices = msr_train.bert_word_embeddings(feature_extraction_model, msr_train.get_flattened_encoded(),
                                                            batch_size)
    paraphrase_embeddings = msr_train.combine_sentence_embeddings(msr_train.aggregate_sentence_embeddings(pairs, inputs,
                                                                                                          indices))

    #code from here on out mostly copied from
    # https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
    learningRate = 0.01
    epochs = 100

    model = LinearRegression(paraphrase_embeddings.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        inputs = Variable(paraphrase_embeddings)
        labels = Variable(labels)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

    torch.save(model.state_dict(), 'test_model.pt')


def test_probe(args):

    msr_test = ParaphraseDataset(args.input, args.embeddings_model, args.embedding_batch_size, args.run_name,
                                 indices=(0, 3, 4)) #TODO: hardcoding MSRP data indices atm
    labels = msr_test.get_labels()
    if args.embeddings_cache is None:
        pairs, inputs, indices = msr_test.bert_word_embeddings(msr_test.get_flattened_encoded())
    else:
        pairs, inputs, indices = msr_test.load_saved_embeddings(args.embeddings_cache)

    paraphrase_embeddings = msr_test.combine_sentence_embeddings(msr_test.aggregate_sentence_embeddings(pairs, inputs,
                                                                                                          indices))

    model = LinearRegression(paraphrase_embeddings.shape[1], 1)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    with torch.no_grad():
        inputs = Variable(paraphrase_embeddings)
        labels = Variable(labels)

        outputs = model(inputs)

        predicted_outputs = torch.squeeze(outputs)

    total = len(predicted_outputs)
    correct = int(torch.sum((torch.round(predicted_outputs) == labels) * 1))
    acc_string = "{}/{} correct for an accuracy of {}".format(correct, total, correct/total)
    output_file(args.run_name, 'classifications.tsv',
                ['\t'.join((str(float(x)), str(int(torch.round(x)))))+'\n' for x in predicted_outputs])
    output_file(args.run_name, 'acc.txt', [acc_string+'\n'])
    module_logger.info(acc_string)

def output_file(run_name, filename, content):
    folder = os.path.join('output', run_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, filename), 'w+') as outfile:
        outfile.writelines(content)


if __name__ == '__main__':
    train, test = 'train', 'test'
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_batch_size', type=int, default=64)
    parser.add_argument('--embeddings_cache', type=str, help='Directory to load cached embeddings from')
    parser.add_argument('--embeddings_model', type=str, default='bert-large-uncased',
                        help='The model used to transform text into word embeddings')
    parser.add_argument('--run', type=str, choices=[train, test], required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='run_{}'.format((int(time.time()))),
                        help='A label for the run, used to name output and cache directories')

    parser.add_argument('--model', type=str, required=True, help='Name of the model')

    args = parser.parse_args()

    if args.run == train:
        train_probe(args)
    elif args.run ==test:
        test_probe(args)





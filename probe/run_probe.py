import json
from torch.autograd import Variable
import torch
from load_data import ParaphraseDataset
import argparse
import logging
import time
import os.path

train, test = 'train', 'test'
combined, cls = 'combined', 'cls'
module_logger = logging.getLogger('probe')

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def train_probe(input_args):
    model_name = input_args.model if input_args.model is not None else 'model.pt'
    train_data = ParaphraseDataset(input_args.input, input_args.embeddings_model, input_args.embedding_batch_size,
                                   input_args.run_name,  indices=(0, 3, 4)) #TODO: hardcoding MSRP data indices atm
    labels = train_data.get_labels()
    if input_args.embeddings_cache is None:
        pairs, inputs, indices = train_data.bert_word_embeddings(train_data.get_flattened_encoded())
    else:
        pairs, inputs, indices = train_data.load_saved_embeddings(input_args.embeddings_cache)

    paraphrase_embeddings = train_data.combine_sentence_embeddings(train_data.aggregate_sentence_embeddings(pairs, inputs,
                                                                                                          indices))

    #code from here on out mostly copied from
    # https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
    learning_rate = input_args.learning_rate
    epochs = input_args.epochs

    model = LinearRegression(paraphrase_embeddings.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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

        module_logger.info('epoch {}, loss {}'.format(epoch, loss.item()))

    eval_model(model, paraphrase_embeddings, labels, input_args)
    output_file(input_args.run_name, 'model_metadata.json', json.dumps({
        'learning_rate': learning_rate,
        'epochs': epochs
    }))
    torch.save(model.state_dict(), os.path.join('output', input_args.run_name, model_name))



def test_probe(input_args):
    test_data = ParaphraseDataset(input_args.input, input_args.embeddings_model, input_args.embedding_batch_size,
                                  input_args.run_name, indices=(0, 3, 4)) #TODO: hardcoding MSRP data indices atm
    labels = test_data.get_labels()

    if input_args.embeddings_cache is None:
        if input_args.embedding_paradigm == combined:
            encoded_data = test_data.get_flattened_encoded()
        else:
            encoded_data = test_data.get_encoded()
        embeddings, inputs, indices = test_data.bert_word_embeddings(encoded_data)
    else:
        embeddings, inputs, indices = test_data.load_saved_embeddings(input_args.embeddings_cache)

    if input_args.embedding_paradigm == combined:
        final_embeddings = test_data.combine_sentence_embeddings(test_data.aggregate_sentence_embeddings(embeddings, inputs,
                                                                                                          indices))
    else:
        final_embeddings = test_data.bert_cls_embeddings(embeddings)

    model = LinearRegression(final_embeddings.shape[1], 1)
    model.load_state_dict(torch.load(input_args.model))

    eval_model(model, final_embeddings, labels, input_args)


def eval_model(model, data, labels, input_args):
    model.eval()
    with torch.no_grad():
        inputs = Variable(data)
        labels = Variable(labels)

        outputs = model(inputs)

        predicted_outputs = torch.squeeze(outputs)

    total = len(predicted_outputs)
    correct = int(torch.sum((torch.round(predicted_outputs) == labels) * 1))
    acc_string = "{}/{} correct for an accuracy of {}".format(correct, total, correct/total)
    output_file(input_args.run_name, '{}_classifications.tsv'.format(input_args.run),
                ['\t'.join((str(float(x)), str(int(torch.round(x)))))+'\n' for x in predicted_outputs])
    output_file(input_args.run_name, '{}_acc.txt'.format(input_args.run), [acc_string + '\n'])
    module_logger.info(acc_string)

def output_file(run_name, filename, content):
    folder = os.path.join('output', run_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, filename), 'w+') as outfile:
        outfile.writelines(content)


def require_args(input_args, required_args):
    for required_arg in required_args:
        if getattr(input_args, required_arg) is None:
            raise Exception("Cannot run {} without --{}".format(input_args.run, required_arg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_batch_size', type=int, default=64)
    parser.add_argument('--embeddings_cache', type=str, help='Directory to load cached embeddings from')
    parser.add_argument('--embeddings_model', type=str, default='bert-large-uncased',
                        help='The model used to transform text into word embeddings')
    parser.add_argument('--embedding_paradigm', type=str, choices=[combined, cls], default=combined,
                        help='Whether to combine sentence embeddings or take the CLS token of joint embeddings')
    parser.add_argument('--run', type=str, choices=[train, test], required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='run_{}'.format((int(time.time()))),
                        help='A label for the run, used to name output and cache directories')

    parser.add_argument('--model', type=str, help='Name of the model')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=0.01)
    parser.add_argument('--epochs', type=int, help='Epochs for training', default=100)

    input_args = parser.parse_args()

    if input_args.run == train:
        train_probe(input_args)
    elif input_args.run ==test:
        require_args(input_args, ['model'])
        test_probe(input_args)





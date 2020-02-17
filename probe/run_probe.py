from transformers import BertTokenizer, BertModel
from torch.autograd import Variable
import torch
from load_data import ParaphraseDataset
import numpy as np

#TODO: this file is mostly still just testcode, please take with a grain of salt

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out



def run_probe():
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    feature_extraction_model = BertModel.from_pretrained('bert-large-uncased')

    batch_size = 20
    msr = ParaphraseDataset('test_head.txt', tokenizer, indices=(0, 3, 4))
    labels = msr.get_labels()
    pairs, inputs, indices = msr.bert_word_embeddings(feature_extraction_model, msr.get_flattened_encoded(), batch_size)
    paraphrase_embeddings = msr.combine_sentence_embeddings(msr.aggregate_sentence_embeddings(pairs, inputs, indices))

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


if __name__ == '__main__':
    run_probe()






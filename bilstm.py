import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchtext import data
from torchtext.vocab import pretrained_aliases, Vocab

from utilities import load_dev, load_train, spacy_tokenizer, LSTMTrainer

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_bilstm(model, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "weights.pth"))

class BiLSTMClassifier(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_size, lstm_hidden_size, classif_hidden_size,
        lstm_layers=1, dropout_rate=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.embed_size = embed_size
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_size, lstm_layers, bidirectional=True, dropout=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, classif_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(classif_hidden_size, num_classes)
        )
    def init_embedding(self, weight):
        self.embedding.weight = nn.Parameter(weight.to(self.embedding.weight.device))
    def forward(self, seq, length):
        seq_size, batch_size = seq.size(0), seq.size(1)
        length_perm = (-length).argsort()
        length_perm_inv = length_perm.argsort()
        seq = torch.gather(seq, 1, length_perm[None, :].expand(seq_size, batch_size))
        length = torch.gather(length, 0, length_perm)
        seq = self.embedding(seq)
        seq = pack_padded_sequence(seq, length)
        features, hidden_states = self.lstm(seq)
        features = pad_packed_sequence(features)[0]
        features = features.view(seq_size, batch_size, 2, -1)
        last_indexes = (length - 1)[None, :, None, None].expand((1, batch_size, 2, features.size(-1)))
        forward_features = torch.gather(features, 0, last_indexes)
        forward_features = forward_features[0, :, 0]
        backward_features = features[0, :, 1]
        features = torch.cat((forward_features, backward_features), -1)
        logits = self.classifier(features)
        logits = torch.gather(logits, 0, length_perm_inv[:, None].expand((batch_size, logits.size(-1))))
        return logits, hidden_states

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument("--output_dir", type=str, default='output', help="Directory where to save the model.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--lr_schedule", type=str, default="warmup", choices=["warmup", "cyclic"],
        help="Schedule to use for the learning rate. Choices are: constant, linear warmup & decay, cyclic.")
    parser.add_argument("--warmup_steps", type=int, default=100,
        help="Warmup steps for the 'warmup' learning rate schedule. Ignored otherwise.")
    parser.add_argument("--epochs_per_cycle", type=int, default=1,
        help="Epochs per cycle for the 'cyclic' learning rate schedule. Ignored otherwise.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=-1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:                                                                                                                                   
        device = torch.device("cpu")
        print("Running on the CPU")
    set_seed(args.seed)
    print("Loading Data")
    train_dataset, text_field = load_train(args.data_dir, spacy_tokenizer)
    valid_dataset = load_dev(args.data_dir, spacy_tokenizer, None)
    out_domain_valid_dataset = load_dev(args.data_dir, spacy_tokenizer,None,'out_dev.tsv')
    vocab = text_field.vocab

    model = BiLSTMClassifier(2, len(vocab.itos), vocab.vectors.shape[-1],lstm_hidden_size=300, classif_hidden_size=400, dropout_rate=0.15).to(device)
    model.init_embedding(vocab.vectors.to(device))
    trainer = LSTMTrainer(model, device,
        loss="cross_entropy",
        train_dataset=train_dataset, val_dataset=valid_dataset, out_val_dataset=out_domain_valid_dataset, val_interval=250,
        checkpt_interval=args.checkpoint_interval,
        checkpt_callback=lambda m, step: save_bilstm(m, os.path.join(args.output_dir, "checkpt_%d" % step)),
        batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr)
    
    if args.train:
        print("Training model")
        trainer.train(args.epochs, schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps, epochs_per_cycle=args.epochs_per_cycle)
    if args.eval:
        print("Evaluating model in domain:{}".format(trainer.evaluate()))
        print("Evaluating model out of domain:{}".format(trainer.evaluate(False)))
    if args.save:
        print("Saving Model")
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        save_bilstm(model, args.output_dir)

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchtext import data
from torchtext.vocab import pretrained_aliases, Vocab
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer)
from utils import set_seed, load_data, BertVocab, BertTrainer, save_bert, save_bilstm, spacy_tokenizer, LSTMTrainer, BiLSTMClassifier

if __name__ == "__main__":
    python model.py --data_dir ../../../data/cola_formated/ --train --evaluate --model bilstm
    python model.py --data_dir ../../../data/cola_formated/ --train --evaluate --model bert
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument("--output_dir", type=str, default="output",  help="Directory where to save the model.")
    parser.add_argument("--cache_dir", type=str, default='.cache', help="Custom cache for transformer models.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--lr_schedule", type=str, default='warmup', choices=["constant", "warmup"])
    parser.add_argument("--warmup_steps", type=int, default=100,)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--load", action="store_true") #implement loading model is a todo
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint_interval", type=int, default=-1)
    parser.add_argument("--model", type=str, default="bilstm", choices["bilstm", "bert", "gpt-2"] )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:{}".format(device))
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        print("Output folder not found. Creating folder")
        os.mkdir(args.output_dir)
    if args.model == "bilstm":
        print("Model choice is BiLSTM. Loading data.")
        train_dataset, valid_dataset, text_field = load_data(args.data_dir, spacy_tokenizer)
        vocab = text_field.vocab
        print("Data Loaded. Initializing classifier")
        model = BiLSTMClassifier(num_classes=2, len(vocab.itos), vocab.vectors.shape[-1], lstm_hidden_size=300, classif_hidden_size=400, dropout_rate=0.0).to(device)
        model.init_embedding(vocab.vectors.to(device)) 
        trainer = LSTMTrainer(model, device,
            loss= "cross_entropy",
            train_dataset=train_dataset, 
            val_dataset=valid_dataset, 
            val_interval=250,
            checkpt_interval=args.checkpoint_interval,
            checkpt_callback=lambda m, step: save_bilstm(m, os.path.join(args.output_dir, "checkpt_%d" % step)),
            batch_size=args.batch_size, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr=args.lr)
    elif args.model == "bert":
        print("Model choice is bert. Loading model and tokenizer.")
        bert_config = BertConfig.from_pretrained("bert-large-uncased", cache_dir=args.cache_dir)
        bert_model = BertForSequenceClassification.from_pretrained("bert-large-uncased", config=bert_config, cache_dir=args.cache_dir).to(device)
        bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True, cache_dir=args.cache_dir)
        print("Bert Model and tokenizer loaded. Loading data.")
        train_dataset, valid_dataset, _ = load_data(args.data_dir, bert_tokenizer.tokenize, vocab=BertVocab(bert_tokenizer.vocab), batch_first=True)
        print("Data Loaded. Initializing classifier")
        trainer = BertTrainer(bert_model, device,
            loss="cross_entropy",
            train_dataset=train_dataset,
            val_dataset=valid_dataset, 
            val_interval=250,
            checkpt_callback=lambda m, step: save_bert(m, bert_tokenizer, bert_config, os.path.join(args.output_dir, "checkpt_%d" % step)),
            checkpt_interval=args.checkpoint_interval,
            batch_size=args.batch_size, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr=args.lr)
    else:
        print("Need to chose one of the model options: bilstm, bert")
        exit(-1)

    if args.train:
        print("Training model for {} epochs. Model has {} learnable params.".format(args.epochs))
        trainer.train(args.epochs, schedule=args.lr_schedule, warmup_steps=args.warmup_steps, epochs_per_cycle=args.epochs_per_cycle)
    if args.evaluate:
        print("Evaluating model")
        print(trainer.evaluate())

    if args.model == "bilstm":
        save_bilstm(model, args.output_dir)
    elif args.model == "bert":
        save_bert(bert_model, bert_tokenizer, bert_config, args.output_dir)
    else:
        exit()
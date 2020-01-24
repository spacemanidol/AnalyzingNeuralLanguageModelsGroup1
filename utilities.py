import os
import re
import csv
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
#import torchvision
#import torchvision.transforms as transforms
#import torch.multiprocessing as mp
#import torch.distributed as dist
from torchtext import data
from torchtext.vocab import pretrained_aliases, Vocab

#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp

import spacy
from spacy.symbols import ORTH

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from tensorboardX import SummaryWriter

from tqdm.autonotebook import tqdm, trange

spacy_en = spacy.load("en")
spacy_en.tokenizer.add_special_case("<mask>", [{ORTH: "<mask>"}])  

class BertVocab:
    UNK = '<unk>'
    def __init__(self, stoi):
        self.stoi = OrderedDict()
        pattern = re.compile(r"\[(.*)\]")
        for s, idx in stoi.items():
            s = s.lower()
            m = pattern.match(s)
            if m:
                content = m.group(1)
                s = "<%s>" % content
            self.stoi[s] = idx
        self.unk_index = self.stoi[BertVocab.UNK]
        self.itos = [(s, idx) for s, idx in self.stoi.items()]
        self.itos.sort(key=lambda x: x[1])
        self.itos = [s for (s, idx) in self.itos]
    def _default_unk_index(self):
        return self.unk_index
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(BertVocab.UNK))
    def __len__(self):
        return len(self.itos)

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

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_tsv(path, skip_header=True):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        if skip_header:
            next(reader)
        data = [row for row in reader]
    return data

def load_dev(data_dir, tokenizer, vocab, dev_file='dev.tsv', batch_first=False):
    text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=batch_first)
    label_field_class = data.Field(sequential=False, use_vocab=False, dtype=torch.long)
    field_valid = [("text", text_field), ("label", label_field_class)]
    valid_dataset = data.TabularDataset(path=os.path.join(data_dir, dev_file),format="tsv", skip_header=True,fields=field_valid)
    if vocab is None:
        vectors = pretrained_aliases['fasttext.en.300d'](cache='.cache/')
        text_field.build_vocab(valid_dataset, vectors=vectors)
    else:
        text_field.vocab = vocab
    return valid_dataset

def load_train(data_dir, tokenizer, vocab=None, batch_first=False ,train_file='train.tsv'):
    text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=batch_first)
    label_field_class = data.Field(sequential=False, use_vocab=False, dtype=torch.long)
    field_train = [("text", text_field), ("label", label_field_class)]
    train_dataset = data.TabularDataset(path=os.path.join(data_dir, train_file),format="tsv",  skip_header=True,fields=field_train)
    # Initialize field's vocabulary
    if vocab is None:
        vectors = pretrained_aliases["fasttext.en.300d"](cache=".cache/")
        text_field.build_vocab(train_dataset, vectors=vectors)
    else:
        # Use bert tokenizer's vocab if supplied
        text_field.vocab = vocab
    return train_dataset, text_field

def get_model_wrapper(model_weights, text_field, device=None):
    if isinstance(text_field, str):
        text_field = torch.load(text_field)
    if isinstance(model_weights, str):
        model_weights = torch.load(model_weights)
    if device is None:
        device = torch.device("cpu")

    vocab = text_field.vocab
    model = BiLSTMClassifier(2, len(vocab.itos), vocab.vectors.shape[-1],
        lstm_hidden_size=300, classif_hidden_size=400, dropout_rate=0.15).to(device)
    model.load_state_dict(model_weights)
    trainer = LSTMTrainer(model, device)
    
    def model_wrapper(text):
        outputs = trainer.infer_one(text, text_field, softmax=True)
        return {
            "Negative": outputs[0],
            "Positive": outputs[1]
        }
    return model_wrapper

class Trainer():
    def __init__(self, model, device,
        loss="cross_entropy",
        train_dataset=None,
        temperature=1.0,
                 val_dataset=None, out_val_dataset=None, val_interval=1,
        checkpt_callback=None, checkpt_interval=1,
        max_grad_norm=1.0, batch_size=64, gradient_accumulation_steps=1,
        lr=5e-5, weight_decay=0.0):
        # Storing
        self.model = model
        self.device = device
        self.loss_option = loss
        self.train_dataset = train_dataset
        self.temperature = temperature
        self.val_dataset = val_dataset
        self.val_interval = val_interval
        self.out_val_dataset = out_val_dataset
        self.checkpt_callback = checkpt_callback
        self.checkpt_interval = checkpt_interval
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        # Initialization
        assert self.loss_option in ["cross_entropy", "mse", "kl_div"]
        if self.loss_option == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        elif self.loss_option == "mse":
            self.loss_function = nn.MSELoss(reduction="sum")
        elif self.loss_option == "kl_div":
            self.loss_function = nn.KLDivLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.train_dataset is not None:
            self.train_it = data.BucketIterator(self.train_dataset, self.batch_size, train=True, sort_key=lambda x: len(x.text), device=self.device)
        else:
            self.train_it = None
        if self.out_val_dataset is not None:
            self.out_val_it = data.BucketIterator(self.out_val_dataset, self.batch_size, train=False, sort_key=lambda x: len(x.text), device=self.device)
        else:
            self.out_val_it = None
        if self.val_dataset is not None:
            self.val_it = data.BucketIterator(self.val_dataset, self.batch_size, train=False, sort_key=lambda x: len(x.text), device=self.device)
        else:
            self.val_it = None
    def get_loss(self, model_output, label, curr_batch_size):
        if self.loss_option in ["cross_entropy", "mse"]:
            loss = self.loss_function(
                model_output,
                label
            ) / curr_batch_size # Mean over batch
        elif self.loss_option == "kl_div":
            # KL Divergence loss needs special care
            # It expects log probabilities for the model's output, and probabilities for the label
            loss = self.loss_function(
                F.log_softmax(model_output / self.temperature, dim=-1),
                F.softmax(label / self.temperature, dim=-1)
            ) / (self.temperature * self.temperature) / curr_batch_size
        return loss
    def train_step(self, batch):
        self.model.train()
        batch, label, curr_batch_size = self.process_batch(batch)
        s_logits = self.model(**batch)[0]
        loss = self.get_loss(s_logits, label, curr_batch_size)
        loss.backward()
        self.training_step += 1
        if self.training_step % self.gradient_accumulation_steps == 0:
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                # Advance learning rate schedule
                self.scheduler.step()
            self.model.zero_grad()
            # Save stats to tensorboard
            self.tb_writer.add_scalar("lr",
                self.scheduler.get_lr()[0] if self.scheduler is not None else self.lr,
                self.global_step)
            self.tb_writer.add_scalar("loss", loss, self.global_step)
            self.global_step += 1
            # Every val_interval steps, evaluate and log stats to tensorboard
            if self.val_interval >= 0 and (self.global_step + 1) % self.val_interval == 0:
                results = self.evaluate()
                print(results)
                for k, v in results.items():
                    self.tb_writer.add_scalar("val_" + k, v, self.global_step)
            # Every checkpt_interval steps, call checkpt_callback to save a checkpoint
            if self.checkpt_interval >= 0 and (self.global_step + 1) % self.checkpt_interval == 0:
                self.checkpt_callback(self.model, self.global_step)
    def train(self, epochs=1, schedule=None, **kwargs):
        # Initialization
        self.global_step = 0
        self.training_step = 0
        self.tb_writer = SummaryWriter()
        steps_per_epoch = len(self.train_dataset) // self.batch_size // self.gradient_accumulation_steps
        total_steps = epochs * steps_per_epoch
        # Initialize the learning rate scheduler if one has been chosen
        assert schedule is None or schedule in ["warmup", "cyclic"]
        if schedule is None:
            self.scheduler = None
            for grp in self.optimizer.param_groups: grp['lr'] = self.lr
        elif schedule == "warmup":
            warmup_steps = kwargs["warmup_steps"]
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr/100, max_lr=self.lr,
                step_size_up=max(1, warmup_steps), step_size_down=(total_steps - warmup_steps),
                cycle_momentum=False)
        elif schedule == "cyclic":
            epochs_per_cycle = kwargs["epochs_per_cycle"]
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr/25, max_lr=self.lr,
            step_size_up=steps_per_epoch // 2,
            cycle_momentum=False)
        # The loop over epochs, batches
        for epoch in trange(epochs, desc="Training"):
            for batch in tqdm(self.train_it, desc="Epoch %d" % epoch):
                self.train_step(batch)
        self.tb_writer.close()
        del self.tb_writer
    def evaluate(self, in_domain=True):
        self.model.eval()
        val_loss = val_accuracy = 0.0
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        to_eval = self.val_it
        if not in_domain:
            to_eval = self.out_val_it
        for batch in tqdm(to_eval, desc="Evaluation"):
            with torch.no_grad():
                batch, label, _ = self.process_batch(batch)
                output = self.model(**batch)[0]
                loss = loss_func(output, label)
                val_loss += loss.item()
                val_accuracy += (output.argmax(dim=-1) == label).sum().item()
        val_loss /= len(to_eval)
        val_accuracy /= len(to_eval)
        return {
            "loss": val_loss,
            "perplexity": np.exp(val_loss),
            "accuracy": val_accuracy
        }
    def infer(self, dataset, softmax=False):
        self.model.eval()
        outputs_idx = 0
        outputs = np.empty(shape=(len(dataset), 2))
        infer_it = data.Iterator(dataset, self.batch_size, train=False, sort=False, device=self.device)
        for batch in tqdm(infer_it, desc="Inference"):
            with torch.no_grad():
                batch, _, batch_size = self.process_batch(batch)
                output = self.model(**batch)[0]
                if softmax:
                    output = F.softmax(output, dim=-1)
                outputs[outputs_idx:outputs_idx + batch_size] = output.detach().cpu().numpy()
                outputs_idx += batch_size
                del output
        return outputs
    def infer_one(self, example, text_field=None, softmax=False):
        self.model.eval()
        if text_field is None:
            text_field = self.train_dataset.fields["text"]
        example = text_field.preprocess(example)
        tokens, length  = text_field.process([example])
        with torch.no_grad():
            batch = self.process_one(tokens, length)
            output = self.model(**batch)[0]
            if softmax:
                output = F.softmax(output, dim=-1)
            output = output.detach().cpu().numpy()
        return output[0]
    def process_batch(self, *args):
        # Implemented by subclasses
        raise NotImplementedError()
    def process_one(self, *args):
        # Implemented by subclasses
        raise NotImplementedError()

class BertTrainer(Trainer):
    def process_batch(self, batch):
        tokens, length = batch.text
        label = batch.label if "label" in batch.__dict__ else None
        length = length.unsqueeze_(1).expand(tokens.size())
        rg = torch.arange(tokens.size(1), device=self.device).unsqueeze_(0).expand(tokens.size())
        attention_mask = (rg < length).type(torch.float32)
        batch = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        return batch, label, tokens.size(0)
    def process_one(self, tokens, length):
        return {
            "input_ids": tokens.to(self.device),
            "attention_mask": torch.ones(tokens.size(), dtype=torch.float32, device=self.device)
        }

class LSTMTrainer(Trainer):
    def process_batch(self, batch):
        tokens, length = batch.text
        label = batch.label if "label" in batch.__dict__ else None
        batch = {
            "seq": tokens,
            "length": length,
        }
        return batch, label, tokens.size(1)
    def process_one(self, tokens, length):
        return {
            "seq": tokens.to(self.device),
            "length": length.to(self.device)
        }

"""
Training a transformer model.

SOURCE: OpenNMT: Open-Source Toolkit for Neural Machine Translation

AUTHORS: Guillaume Klein
         Yoon Kim
         Yuntian Deng
         Jean Senellart
         Alexander M. Rush

EDITORS:  Luke Nonas-Hunter
"""

import os
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchtext import data
from transformer import *


class Training:
    """
    Contains all functions for model training.

    Instantiating this class will train a model on the given dataset and save
    the parameters in the given file name.
    """

    def __init__(self, dataset, model_file):
        """
        Train a model on the given dataset and save the resulting parameters.

        Args:
            dataset: Filename of dataset (including .csv).
            model_file: Name of file where parameters will be saved.
        Assumptions:
            dataset files are located in the data folder.
            model files are located in the model folder.
        """
        dataset = AdventureDataset(f"data/{dataset}")
        pad_idx = dataset.vocab.stoi["<PAD>"]
        loader = DataLoader(dataset=dataset,
                            batch_size=32,
                            num_workers=8,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=AdventureCollate(pad_idx=pad_idx))
        batch_iter = []
        for (src, tgt) in loader:
            print(f"SRC: {src}")
            print(f"TGT: {tgt}")
            batch_iter.append(Batch.rebatch(0, src, tgt))

        model = Transformer.make_model(len(dataset.vocab),
                                       len(dataset.vocab))
        criterion = LabelSmoothing(size=len(dataset.vocab),
                                   padding_idx=pad_idx,
                                   smoothing=0.1)
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(),
                                             lr=0,
                                             betas=(0.9, 0.98),
                                             eps=1e-9))
        for epoch in range(50):
            model.train()
            Training.run_epoch(batch_iter, model,
                               SimpleLossCompute(model.generator,
                                                 criterion,
                                                 model_opt))
            model.eval()
            print(Training.run_epoch(batch_iter, model,
                                     SimpleLossCompute(model.generator,
                                                       criterion,
                                                       None)))
        for i, batch in enumerate(batch_iter):
            src = batch.src[:1]
            src_mask = (src != dataset.vocab.stoi["<PAD>"]).unsqueeze(-2)
            out = Transformer.greedy_decode(model,
                                            src,
                                            src_mask,
                                            max_len=60,
                                            start_symbol=dataset
                                            .vocab.stoi["<SOS>"])
            print(f"Source Mask: {src_mask}")
            print(f"Source: {src}")
            print(f"Target: {batch.trg.data[0,:]}")
            print("Output:", end="\t")
            for i in range(1, out.size(1)):
                sym = dataset.vocab.itos[out[0, i].item()]
                if sym == "<EOS>":
                    break
                print(sym, end=" ")
            print()
            print("Target:", end="\t")
            for i in range(1, batch.trg.size(1)):
                sym = dataset.vocab.itos[batch.trg.data[0, i].item()]
                if sym == "<EOS>":
                    break
                print(sym, end=" ")
            print()
            break
        torch.save(model.state_dict(), f"./model/{model_file}")
        print("#### Training Complete ####")
        print("####    Model Saved    ####")

    @staticmethod
    def run_epoch(data_iter, model, loss_compute):
        """
        Helper: Standard training and logging function.

        Args:
            data_iter: List of Batch instances containing data to be
                run through the algorithm.
            model: Instance of the Transformer class.
                Represents transformer model.
            loss_compute: Function to compute loss or "incorrectness".
        """
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        enumerated_data = enumerate(data_iter)
        for i, batch in enumerated_data:
            out = model.forward(batch.src, batch.trg,
                                batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 50 == 1:
                elapsed = time.time() - start
                print(f"Epoch Step: {i} \
                        Loss: {loss / batch.ntokens} \
                        Tokens per Sec: {tokens / elapsed}")
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    @staticmethod
    def get_std_opt(model):
        """
        Helper: Create a standard optimization function.

        Args:
            model: Instance of Transformer class which will use the
                optimization function.
        Returns:
            A standard optimization function based on values determined by
            the team working on OpenNMT.
        """
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(),
                                        lr=0,
                                        betas=(0.9, 0.98),
                                        eps=1e-9))

    @staticmethod
    def data_gen(V, batch, number_batches):
        """
        Helper: Generate random data for a src-tgt copy task.

        Args:
            V: Integer size of data to be generated.
            batch: Integer size of batches.
            number_batches: Integer number of batches.
        """
        for i in range(number_batches):
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            data[:, 0] = 1
            src = Variable(data, requires_grad=False)
            tgt = Variable(data, requires_grad=False)
            yield Batch(src, tgt, 0)


class Batch:
    """
    Object for holding a batch of data with mask during training.

    Attributes:
        src: List of source data.
        src_mask: List representing how much of the source data to mask at any
            given time.
        trg: List of target data.
        trg_mask: List representing how much of the target data to mask at any
            given time.
        ntokens: Integer number of token words in batch.
    """

    def __init__(self, src, trg=None, pad=0):
        """
        Instantiate a Batch of data containing the source data and target data.

        Args:
            src: List of source data.
            trg: List of target data.
            pad: Integer amount to pad the data with.
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Helper: Create a mask to hide padding and future words.

        Args:
            tgt: List of tokens.
            pad: Integer amount to describe how much data the list was padded
                with.
        Returns:
            A list of booleans with the same length as the target list which
            represents what words the model should address at any given moment.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            Transformer.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    @staticmethod
    def rebatch(pad_idx, src_in, trg_in):
        """
        Helper: Transpose tensors from loaded datasets so they have the proper
        dimensions.

        Args:
            pad_idx: Integer amount of data to pad original data with.
            src_in: List of source data.
            trg_in: List of target data.
        """
        src, trg = src_in.transpose(0, 1), trg_in.transpose(0, 1)
        return Batch(src, trg, pad_idx)


class NoamOpt:
    """
    Optimizer function wrapper that implements rate.

    Currently, this model uses the Adam Optimizer. This is subject to change
    in the future.

    Attributes:
        optimizer: Optimizer function.
        _step: Integer step size for each itteration.
        warmup: Integer warmup step amount. This describes how many steps
            before the rate switches from linearly increasing to increasing
            proportionally with the inverse square root.
        factor: Float factor to multiply with output of function.
        model_size: Integer dimension of model input layer.
        _rate: Rate at which optimizer function should change. Rate changes
            as _step increases.
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Instantiates an optimizer wrapper.

        Args:
            model_size: Integer dimension of model input layer.
            factor: Float factor to multiply with outut of optimizer function.
            warmup: Integer of steps to be considered apart of the warmup.
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        Update parameters and rate.
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement 'lrate' above

        Args:
            step: Integer step amount used to override saved step amount.
        Returns:
            The rate to be used with the optimizer function based on current
            step.
        """
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5),
             step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing to reduce model confidence while training.

    Attributes:
        criterion: Function which decides if a node is producing the correct
            response.
        padding_idx: Integer representation of <PAD> token.
        confidence: Float representing the models confidence in its answers.
        smoothing: Float representing amount of smoothing applied.
        size: Integer size of the model.
        true_dist: The distance between the correct answer and the given data.
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        Instantiate a LabelSmoothing object.

        Args:
            size: Integer size of the model.
            padding_idx: Integer representation of <PAD> token.
            smoothing: Float representing amount of smoothing applied.
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        Feed data forward through LabelSmoothing layer.

        For use with the pytorch library.

        Args:
            x: Input data.
            target: Expected output data.
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    """
    A simple loss compute and train function.

    Attributes:
        generator: Instance of the generator class defined in the model.
        criterion: Instance of LabelSmoothing class.
        opt: Instance of NoamOpt class.
    """

    def __init__(self, generator, criterion, opt=None):
        """
        Create instance of SimpleLossCompute class.

        Args:
            generator: Instance of the generator class defined in the model.
            criterion: Instance of LabelSmoothing class.
            opt: Instance of NoamOpt class.
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        Calculates the loss (or incorrectness) of data.

        Args:
            x: Source data.
            y: Target data.
            norm: Normalization factor
        """
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()

            self.opt.optimizer.zero_grad()
        return loss.data * norm


class Vocabulary:
    """
    Class containing the models vocabulary based on given dataset.

    Attributes:
        itos: Dictionary which translates indecies to tokens
        stoi: Dictionary which translates tokens to indecies
        freq_threshold: Integer representing amount of times a word needs to
            appear in the dataset for it to become a token.
        spacey_en: Sentence tokenizer defined in spacy package.
    """

    spacy_en = spacy.load('en')

    def __init__(self, freq_threshold):
        """
        Instantiate a vocabulary object.

        Args:
            freq_threshold: Integer representing amount of times a word needs
                to appear in the dataset for it to become a token.
        """
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        """
        Returns the length of tokens/indecies in the vocabulary.
        """
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        """
        Defines the itos and stoi based on a list of given sentences.

        NOTE: This function may be replaced by a different function
        which defines the vocbulary by using word grouping on a more
        general set of vocabulary words. This will help increase the
        model's accurary.

        Args:
            sentence_list: List of strings containing words to load into the
                model. These sentences usually come from the dataset the model
                was trained on.
        """
        frequenecies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in frequenecies:
                    frequenecies[word] = 1
                else:
                    frequenecies[word] += 1

                if frequenecies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text, tokenize=True):
        """
        Convert a list of tokens (or a string) into a list of their respective
        indecies.

        Args:
            text: A list of tokens or a string.
            tokenize: A boolean determining if the text parameter needs to be
                tokenized.
        Returns:
            A list of indicies representing the given text.
        """
        if tokenize:
            text = self.tokenize(text)

        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in text]

    @staticmethod
    def tokenize(text):
        """
        Separate a string of text into tokens using the spacy tokenizer.

        Args:
            text: String of english text.
        Returns:
            A list of strings, refered to as tokens, that were created from the
            original string.
        """
        return [tok.text for tok in Vocabulary.spacy_en.tokenizer(text)]


class AdventureDataset(Dataset):
    """
    Represents a dataset of text used to train the model and translate strings
    into numerical representations.

    Attributes:
        root_dir: String containing the root directory of the project.
        data_frame: Pandas data frame containg dataset read from a csv.
        transform: Function indicating how data from dataframe needs to be
            transformed.
        src: Source data.
        tgt: Target data.
        vocab: Vocabulary object containing all vocabulary from the dataset
            that appears atleast as many times of the given frequency
            threshold.
    """

    def __init__(self, data_file, root_dir="./",
                 transform=None, freq_threshold=1):
        """
        Instantiate AdventureDataset object.

        Args:
            data_file: String containing path to csv file containing training
                data.
            root_dir: String containing root directory of project.
            transform: Function indicating how data from dataframe needs to be
                transformed.
            freq_threshold: Integer number of times a word needs to appear in
                the dataset to be converted into a token.
        """
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(data_file)
        self.transform = transform

        # Get source, target
        self.src = self.data_frame["source"]
        self.tgt = self.data_frame["target"]
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.src.tolist() + self.tgt.tolist())

    def __len__(self):
        """
        Return length of training data.
        """
        return len(self.data_frame)

    def __getitem__(self, index):
        """
        Return the source-target vectors for the given training data at a
            specified index.

        Args:
            index: Integer index of the dataset.
        Returns:
            Numericalized and tokenized source-target data pair from the
            dataset.
        """
        source = self.src[index]
        target = self.tgt[index]

        if self.transform is not None:
            # INSERT TEXT TRANSFORMATION HERE
            pass

        numericalized_source = [self.vocab.stoi["<SOS>"]]
        numericalized_source += self.vocab.numericalize(source)
        numericalized_source.append(self.vocab.stoi["<EOS>"])

        numericalized_target = [self.vocab.stoi["<SOS>"]]
        numericalized_target += self.vocab.numericalize(target)
        numericalized_target.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_source), \
            torch.tensor(numericalized_target)


class AdventureCollate:
    """
    Restructures batches so that all batches are the same length.

    Attributes:
        pad_idx: Integer index representing the <PAD> token.
    """

    def __init__(self, pad_idx):
        """
        Instantiates AdventureCollate class and sets the index of the <PAD>
        token.

        Args:
            pad_idx: Integer index representing the <PAD> token.
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Restructures batches so they are all roughly the same size.

        Args:
            batch: A collection of source and target data.
        Returns:
            The restructured source and target data.
        """
        source = [item[0] for item in batch]
        source = pad_sequence(source,
                              batch_first=False,
                              padding_value=self.pad_idx)
        target = [item[1] for item in batch]
        target = pad_sequence(target,
                              batch_first=False,
                              padding_value=self.pad_idx)
        padding_difference = target.shape[0] - source.shape[0]
        if padding_difference > 0:
            source = F.pad(input=source, pad=(0, 0, 0, padding_difference))
        if padding_difference < 0:
            target = F.pad(input=target, pad=(0, 0, -padding_difference, 0))
        return source, target


if __name__ == "__main__":
    training = Training("test.csv", "italian_numbers")

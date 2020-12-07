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

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformer import *

class Training:
    """
    Contains all functions for model training.
    """

    @staticmethod
    def run_epoch(data_iter, model, loss_compute):
        """
        Standard training and logging function.
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
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(),
                                        lr=0,
                                        betas=(0.9, 0.98),
                                        eps=1e-9))

    @staticmethod
    def data_gen(V, batch, nbatches):
        """
        Generate random data for a src-tgt copy task.
        """
        for i in range(nbatches):
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            data[:, 0] = 1
            src = Variable(data, requires_grad=False)
            tgt = Variable(data, requires_grad=False)
            yield Batch(src, tgt, 0)

class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    max_src_in_batch = None
    max_tgt_in_batch = None

    def __init__(self, src, trg=None, pad=0):
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
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            Transformer.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    @staticmethod
    def batch_size_fn(self, new, count, sofar):
        """
        Keep augmenting batch and calculate total number of tokens + padding.
        """
        if count == 1:
            self.max_src_in_batch = 0
            self.max_tgt_in_batch = 0
        self.max_src_in_batch = max(self.max_src_in_batch, len(new.src))
        self.max_tgt_in_batch = max(self.max_tgt_in_batch, len(new.trg) + 2)
        src_elements = count * self.max_src_in_batch
        tgt_elements = count * self.max_tgt_in_batch
        return max(src_elements, tgt_elements)

class NoamOpt:
    """
    Optim wrapper that implements rate.
    """

    def __init__(self, model_size, factor, warmup, optimizer):
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
        """
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class LabelSmoothing(nn.Module):
    """
    Implement label smoothing.
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
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
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()

            self.opt.optimizer.zero_grad()
        return loss.data * norm

if __name__ == "__main__":
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = Transformer.make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        Training.run_epoch(Training.data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(Training.run_epoch(Training.data_gen(V, 30, 5), model,
                                 SimpleLossCompute(model.generator,
                                                   criterion,
                                                   None)))

    # Run and decode model copy
    model.eval()
    src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(Transformer.greedy_decode(model,
                                    src,
                                    src_mask,
                                    max_len=10,
                                    start_symbol=1))

"""
Implementation of a language predictor based on a transformer architecture
by the wonderful OpenNMT team at Harvard.

SOURCE: OpenNMT: Open-Source Toolkit for Neural Machine Translation

AUTHORS: Guillaume Klein
         Yoon Kim
         Yuntian Deng
         Jean Senellart
         Alexander M. Rush

EDITORS:  Luke Nonas-Hunter
"""

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time

class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.

    Attributes:
        encoder:
        decoder:
        src_embed:
        tgt_embed:
        generator:
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Instantiate the model.
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask),
                           src_mask,
                           tgt,
                           tgt_mask)

    def encode(self, src, src_mask):
        """
        Run values through the encoder.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Run values through the decoder.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @staticmethod
    def clones(self, module, N):
        """
        Produce N identifical layers.

        Args:
            module: Module to be duplicated.
            N: Integer number of models to duplicate.
        """
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    @staticmethod
    def subsequent_mask(size):
        """
        Mask out subsequent positions
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    @staticmethod
    def make_model(src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.
    """
    c= copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoerLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderyLayer(d_model, c(attn), c(attn),
                              c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nnSequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab):
        """
        Instantiate standard linear + softmax neural network.
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Feed data through the network.

        Args:
            x: The data to be passed through the network.
        """
        return F.log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.
    """
    
    def __init__(self, layer, N):
        """
        Instantiate encoder module.
        """
        super(Encoder, self).__init__()
        self.layers = Transformer.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each later in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """
    Construct a normalization layer.
    """
    
    def __init__(self, features, eps=1e-6):
        """
        Instantiate layer norm.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        Feed data through the layer.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        """
        Initalize a sublayer connection
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    Assemble an encoder layer for use inside an encoder.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = Transformer.clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feedforward
    """
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayerm self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    """
    Combines multiple attention functions together into one layer.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implementation of multiheaded attention
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [1(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for 1, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                    dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous() \ 
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """
    Implementats FFN equation."
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class Embeddings(nn.Module):
    """
    Embed text as vectors
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)p

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.sze(1)],
                         requires_grad=False)
        return self.dropout(x)

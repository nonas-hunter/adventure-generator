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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import math
import copy
import time


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder transformer architecture.

    Attributes:
        encoder: Instance of the encoder module.
        decoder: Instance of the decoder module.
        src_embed: Instance of embeddings module for the source data.
        tgt_embed: Instance of embeddings module for the target data.
        generator: Instance of generator module.
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

        Args:
            src: Source data tensor.
            tgt: Target data tensor.
            src_mask: Boolean tensor illustrating what source data should be
                examined at any given timestep.
            tgt_mask: Boolean tensor illustrating what target data should be
                examined at any given timestep.
        Returns:
            Output of the decode element given the output of the encode element
            and the target data (or memory).
        """
        return self.decode(self.encode(src, src_mask),
                           src_mask,
                           tgt,
                           tgt_mask)

    def encode(self, src, src_mask):
        """
        Run values through the encoder module.

        Args:
            src: Source tensor to run through encoder module.
            src_mask: Boolean tensor indicating what data should be examined
                at any given timestep.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Run values through the decoder.

        Args:
            memory: Data tensor that contains previous words predicted by the
                model.
            src_mask: Boolean tensor indicating what source data should be
                examined at any given timestep.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def generate_text(self, tokenized_prompt, vocab):
        """
        Generate text given a text prompt and vocabulary.

        Args:
            tokenized_prompt: List of ordered words from the input sentence.
            vocab: Instance of vocabulary object containing all vocabulary from
                the dataset the model was trained on.
        """
        assert len(tokenized_prompt) <= 510
        numerical_prompt = [torch.tensor([vocab.stoi["<SOS>"]]
                                         + vocab.numericalize(tokenized_prompt,
                                                              tokenize=False)
                                         + [vocab.stoi["<EOS>"]]),
                            torch.tensor([0 for _
                                          in range(512)])]
        numerical_prompt = pad_sequence(numerical_prompt,
                                        batch_first=False,
                                        padding_value=0)
        numerical_prompt = numerical_prompt.transpose(0, 1)
        mask = (numerical_prompt[0] != vocab.stoi["<PAD>"]).unsqueeze(-2)
        model_out = self.greedy_decode(self,
                                       numerical_prompt[0], mask,
                                       max_len=256,
                                       start_symbol=vocab.stoi["<SOS>"],
                                       end_symbol=vocab.stoi["<EOS>"])
        text = ""
        for i in range(1, model_out.size(1)):
            sym = vocab.itos[model_out[0, i].item()]
            if sym == "<EOS>":
                break
            text += f"{sym} "
        return text

    @staticmethod
    def clones(module, N):
        """
        Helper: Produce N identical layers.

        Args:
            module: Module to be duplicated.
            N: Integer number of models to duplicate.

        Returns:
            N identical layers.
        """
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    @staticmethod
    def subsequent_mask(size):
        """
        Helper: Mask out subsequent positions

        Args:
            size: Size of model.
        Returns:
            Tensor representing new mask.
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    @staticmethod
    def make_model(src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        """
        Helper: Construct a model from hyperparameters.

        d_model and d_ff must be divisable by h

        Args:
            src_vocab: Integer number of vocabulary words used in the source
                sentences.
            tgt_vocab: Integer number of vocabulary words used in the target
                sentences.
            N: Integer of decode and encode layers.
            d_model: Integer size of model (determines length of input).
            d_ff: Integer size of the feed forward network.
            h: The number of attention heads.
            dropout: Float representing the rate of node deactivation.
        Return:
            An instance of the Transformer class based on the structure
            outlined by the parameters of the function.
        """
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = Transformer(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Helper: Compute 'Scaled Dot Product Attention'

        Args:
            query: Tensor containing query data.
            key: Tensor containing key data.
            value: Tensor containing value data.
            mask: Boolean tensor illustrating what data should be examined.
            dropout: Float representing the rate of node deactivation.
        Return:
            An output tensor of a single attention head.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    @staticmethod
    def greedy_decode(model, src, src_mask, max_len,
                      start_symbol, end_symbol=None):
        """
        Helper: Calculate the best word given the model output.

        Args:
            model: Instance of the Transformer class.
            src: Source data (also refered to as the prompt).
            src_mask: Boolean list illustrating what data should be examined.
            max_len: Maximum tokens in output
            start_symbol: Token indicating start of sentence.
            end_symbol: Token indicating end of sentence.
        Return:
            A tensor containing the index representation of the output
            sentence generated by the model given the original prompt.
        """
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask,
                               Variable(ys),
                               Variable(Transformer.subsequent_mask(ys.size(1))
                                        .type_as(src.data)))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys,
                            torch.ones(1, 1)
                            .type_as(src.data)
                            .fill_(next_word)], dim=1)
            if end_symbol is not None and next_word == end_symbol:
                break
        return ys


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.

    Attributes:
        proj: A linear feed forward module which takes the output of the decode
            block and projects it onto a series of nodes which represents all
            the possible vocabulary words the model knows.
    """

    def __init__(self, d_model, vocab):
        """
        Instantiate standard linear + softmax neural network.

        Args:
            d_model: Integer dimension of model input.
            vocab: Integer size of vocabulary
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Feed data through the layer.

        Args:
            x: The data to be passed through the network.
        Returns:
            The output of the self.proj layer after being processed by a
            softmax function.
        """
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.

    Attributes:
        layers: List of encoder layers which make up the module.
        norm: Instance of LayerNorm.
    """

    def __init__(self, layer, N):
        """
        Instantiate encoder module.

        Args:
            layer: Instance of EncoderLayer to be duplicated.
            N: Integer of times to duplicate the given layer.
        """
        super(Encoder, self).__init__()
        self.layers = Transformer.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each later in turn.

        Args:
            x: Source data tensor.
            mask: Boolean tensor illustrating what data should be examined.
        Returns:
            Output of all the encoder layers.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Construct a normalization layer.

    Attributes:
        a_2: Parameter tensor which represents the constant that scales the
            output of the normalization function.
        b_2: Parameter tensor which represents the constant that offsets the
            output of the normalization function.
        eps: Float point representing learning rate.
    """

    def __init__(self, features, eps=1e-6):
        """
        Instantiate layer norm.

        Args:
            features: Integer number of features in layer.
            eps: Floating point representation of learning rate.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        Feed data through the layer.

        Args:
            x: Source data tensor.
        Returns:
            Normalized data.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.

    Attributes:
        norm: Instance of LayerNorm.
        dropout: Dropout function defined by pytorch.
    """

    def __init__(self, size, dropout):
        """
        Initalize a sublayer connection.

        Args:
            size: Size of the layer.
            dropout: Floating point representation of dropout rate.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with same size.

        Args:
            x: Source tensor.
            sublayer: Previous layer to run data through.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Assemble an encoder layer for use inside an encoder.

    Attributes:
        self_attn: Instance of the MultiHeadedAttention class.
        feed_forward: Linear feed forward neural network.
        sublayer: List of SublayerConnection modules which normalize the data.
        size: Size of the model.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Instantiate an encoder layer.

        Args:
            size: Integer size of the layer.
            self_attn: Instance of the MultiHeadedAttention class.
            feed_forward: Linear feed forward neural network.
            dropout: Floating point representation of dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = Transformer.clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Feed data through layer.

        Args:
            x: Source data tensor.
            mask: Boolean tensor illustrating what data should be examined.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.

    Attributes:
        layers: List of decoder layers which make up the module.
        norm: Instance of LayerNorm.
    """

    def __init__(self, layer, N):
        """
        Instantiate a decoder module.

        Args:
            layer: Instance of DecoderLayer to be duplicated.
            N: Integer of times to duplicate the given layer.
        """
        super(Decoder, self).__init__()
        self.layers = Transformer.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Feed data through the decoder.

        This includes running the data through all the decoder layers and the
        normalization later.

        Args:
            x: Source data tensor.
            memory: Tensor containing words generated by the model so far.
            src_mask: Boolean tensor illustrating what source data should be
                examined.
            tgt_mask: Boolean tensor illustrating what memory data should be
                examined.
        Returns:
            Output from the decoder.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder layer is made of self-attn, src-attn, and feed forward.

    Attributes:
        size: Integer dimension of model.
        self_attn: Instance of the MultiHeadedAttention class.
        src_attn: Instance of the MultiHeadedAttention class.
        feed_forward: Linear feed forward neural network.
        sublayer: List of connecting sublayers.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Instantiate a decoder layer.

        Args:
            size: Integer dimension of model.
            self_attn: Instance of the MultiHeadedAttention class.
            src_attn: Instance of the MultiHeadedAttention class.
            feed_forward: Linear feed forward neural network.
            dropout: Floating point representation of dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = Transformer.clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Feeds data through decoder layer.

        Args:
            x: Source data tensor.
            memory: Tensor containing words generated by the model so far.
            src_mask: Boolean tensor illustrating what source data should be
                examined.
            tgt_mask: Boolean tensor illustrating what memory data should be
                examined.
        Returns:
            Output of a single decoder layer.
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """
    Combines multiple attention functions together into one layer.

    Attributes:
        d_k: Integer dimension of query, key, and value vectors.
        h: Integer number of attention heads.
        linears: List of linear feed forward layers.
        attn: Function representing a single attention head.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads and instantiate
        MultiHeadedAttention object.

        Args:
            h: Integer number of attention heads.
            d_model: Integer dimension of model.
            dropout: Floating point representation of dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = Transformer.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Feed data through a multiheaded attention layer.

        Args:
            query: Data tensor of query words.
            key: Data tensor of key words.
            value: Data tensor of value words.
            mask: Boolean tensor which indicates which data should be excluded.
        Returns:
            Output of multiheaded attention layer.
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = Transformer.attention(query, key, value, mask=mask,
                                             dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Implementats FFN.

    Attributes:
        w_1: First liner layer in the network.
        w_2 Second linear layer in the network
        dropout: Dropout function defined by pytorch.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Instantiate PositionwiseFeedForward object.

        d_model: Integer dimension of the input/output layers.
        d_ff: Integer dimension of the hidden layers.
        dropout: Floating point representation of dropout layer.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Feed data through the network.

        Args:
            x: Source data tensor.
        Returns:
            Output of the feed forward network.
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    Embed text as vectors.

    Attributes:
        lut: A look up table defined by the pytorch Embedding module.
        d_model: Integer dimension of the model.
    """

    def __init__(self, d_model, vocab):
        """
        Instantiates an embedding module.

        d_model: Integer dimension of the model.
        vocab: Integer number of vocabulary words.
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Feed data through the embedding module.

        Args:
            x: Source data tensor.
        Returns:
            Output of embeddings layer.
        """
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.

    Attributes:
        dropout: Dropout function defined by pytorch.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        """
        Instantiate positional encoding module.

        Args:
            d_model: Integer dimension of model.
            dropout: Floating point representation of dropout rate.
            max_len: Maximum position value.
        """
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
        """
        Feed data through the positional encoding layer.

        Args:
            x: Source data tensor.
        Returns:
            The source data now encoded with each words relative position in
            the sentence.
        """
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":
    tmp_model = Transformer.make_model(10, 10, 2)

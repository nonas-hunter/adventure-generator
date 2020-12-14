# The Dungeons
### Luke Nonas-Hunter
An exploration into AI-generated choose your own adventure stories. This project is heavily inspired by [AI Dungeons](https://play.aidungeon.io), an AI based choose your own adventure game created by Latitude. The goal of this project is to build off of the gameplay in AI Dungeons and add other features such as inventory, health, and narrative consistency.

## The Architecture
The architecture for this project is losely based off of an MVC architecture. A model class keeps track of game state. It updates the game state using a transformer based recurrent neural network to generate text given the previous game state. The neural network is contained in a separate class to make it easier to alter the text generation without affecting the rest of the game. The other two classes, view and controller, are very simple. The view class prints text to the terminal while the controller prompts the user to input text into the terminal. The reason these classes are kept seperate despite their simplicity is that they will most likely represent two different components in the final version of the game which will hopefully be hosted on the website. A simple class structure is defined bellow.

```text
└── Adventure                                       [Main class]
    └── View                                        [Displays game to user]
    └── Controller                                  [Gather input from user]
    └── Model                                       [Keeps track of game state]
        └── Transformer                             [Pytorch transformer language model]
            └── Generator                           [Standard linear layer + softmax to generate output]
            └── Embeddings                          [Embed text as vector]
                └── PositionalEncoding              [Encode position data in input vector]
            └── Encoder                             [Stack of encoder layers]
                └── EncoderLayer                    [A single encoder layer]
                    └── SublayerConnection          [Intermidate connection before a LayerNorm]
                        └── LayerNorm               [A normilization layer]
                    └── MultiHeadedAttention        [Contains multiple attention functions]
                    └── PositionwiseFeedForward     [Feed forward network]
                └── LayerNorm                       [A normilization layer]
            └── Decoder                             [Stack of decoder layers]
                └── DecoderLayer                    [A single decoder layer]
                    └── SublayerConnection          [Intermidate connection before a LayerNorm]
                        └── LayerNorm               [A normilization layer]
                    └── MultiHeadedAttention        [Contains multiple attention functions]
                    └── PositionwiseFeedForward     [Feed forward network]
                └── LayerNorm                       [A normilization layer]
        └── Training                                [Train the language model]
            └── Batch                               [Holds batches of source and target data for training]
            └── NoamOpt                             [Optim wrapper that implements changing rate]
            └── LabelSmoothing                      [Reduce model's confidence]
            └── SimpleLossCompute                   [Calculate incorrectness of model for training]
            └── AdventureCollate                    [Restructure batches for use in training]
            └── AdventureDataset                    [Load training and vocab data from csv]
                └── Vocabulary                      [Hold model vocabulary]
```

## Running the Code
To run the code you will need the following packages:

- [pytorch](https://pytorch.org/get-started/locally/)
- [spacey](https://spacy.io/usage)
- [numpy](https://numpy.org/install/)

Clicking any of the names above will bring you to the relavent instrucitions to install the package for your given platform.

Running the python file adventures.py should begin a basic adventure in the terminal.

## TODO
- Create website
- Create presentation
- Submit github link
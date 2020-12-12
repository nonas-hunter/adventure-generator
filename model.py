from transformer import *
from training import *


class Model:
    """
    Keeps track of the Adventure state.

    Attributes:
        _vocab: A dataset containing all the vocabulary the network was
            trained on.
        _ai: An instance of a transformer neural network used for text
            generation.
    """

    def __init__(self, model_file, data_file):
        """
        Instanitates game vairables and language model.

        Args:
            model_path: String containing path to file containing pretrained
                model parameters.
        """
        self._vocab = AdventureDataset("./", f"data/{data_file}").vocab
        self._text = []
        self._ai = Transformer.make_model(len(self._vocab),
                                          len(self._vocab), N=6)
        self._ai.load_state_dict(torch.load(f"./model/{model_file}"))

    def generate_text(self, prompt):
        """
        Use the neural network to generate text.

        Args:
            prompt: String which contains prompt for model to expand apon.
        """
        tokenize_text = self._vocab.tokenize(prompt)
        self._text.append(tokenize_text)
        if len(self._text) > 1022:
            self._text = self._text[len(self._text) - 1022:]
        return self._ai.generate_text(prompt, self._vocab)


if __name__ == "__main__":
    model = Model("TEST", "data_TRAIN.csv")
    print(model.generate_text("Hello my name is "))

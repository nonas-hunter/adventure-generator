from text_generator import TextGenerator

class Model:
    """
    Keeps track of the Adventure state.

    Attributes:
        _health: An integer representing the players health.
        _ai: An instance of a transformer neural network used for text
            generation.
    """

    def __init__(self):
        """
        Instanitates model and associated neural network.
        """
        self._ai = TextGenerator()

    def generate_text(self):
        """
        Use the neural network to generate text.
        """
        pass

    def update(self):
        """
        Update the game state.
        """
        pass

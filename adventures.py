from model import Model
from view import View
from controller import Controller

class Adventure:
    """
    An implementation of an AI generated choose your own adventure story.

    Attributes:
        _model: An instance of the model class which keeps track of game state.
        _view: An instance of the view class which displays the game to the
            user.
        _controller: An instance of the controller class which gathers user
            input.
    """

    def __init__(self):
        """
        Instanitate necessary variables for the game.
        """
        self._model = Model()
        self._view = View()
        self._controller = Controller()

    def run(self):
        """
        Runs the main loop of the game.
        """
        # Generate text         generated_text = _model.generate_text()
        # Update game state     _model.update()
        # Display text          _view.display(generated_text)
        # Get user input        user_input = _controller.prompt()
        # REPEAT!
        pass

if __name__ == "__main__":
    game = Adventure()
    game.run()

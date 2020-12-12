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
        self._model = Model("TEST", "data_TRAIN.csv")
        self._view = View()
        self._controller = Controller()

    def run(self, prompt):
        """
        Runs the main loop of the game.

        Args:
            prompt: String used as a prompt for the AI to start the game.
        """
        game_over = False
        while not game_over:
            # Display prompt
            self._view.display(prompt, new_line=True)

            # Generate text
            output = self._model.generate_text(prompt)

            # Display text
            self._view.display(output)

            # Get user input
            prompt = self._controller.prompt()

            # Check if game is over
            if prompt == "EXIT":        # User exit
                game_over = True


if __name__ == "__main__":
    game = Adventure()
    game.run("Hello my name is ")

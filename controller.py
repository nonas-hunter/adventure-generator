class Controller:
    """
    Collects user input.

    NOTE: In its current state I don't think this class should be a class.
    It would make a lot more sense as just a function however, I expect
    its complexity to increase when a final platform is decided for the
    application. Therefore I am keeping this class in order to make it
    easier to increase its complexity later down the road.
    """

    def __init__(self):
        """
        Instantiate the input class
        """
        pass

    def prompt(self):
        """
        Prompt the user to input text.

        Returns:
            A string containing user input.
        """
        return input("[INPUT]: ")

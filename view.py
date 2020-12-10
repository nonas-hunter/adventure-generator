class View:
    """
    Displays adventure to the user.

    NOTE: In its current state I don't think this class should be a class.
    It would make a lot more sense as just a function however, I expect
    its complexity to increase when a final platform is decided for the
    application. Therefore I am keeping this class in order to make it
    easier to increase its complexity later down the road.
    """

    def __init__(self):
        """
        Instantiate display class.
        """
        pass

    def display(self, text, new_line=False):
        """
        Display the given text.

        Args:
            text: A string containing text to be displayed to user.
        """
        if new_line:
            print()
        print(text)
        if new_line:
            print()

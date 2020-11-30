# The Dungeons
### Luke Nonas-Hunter
An exploration into AI-generated choose your own adventure stories. This project is heavily inspired by [AI Dungeons](play.aidungeon.io), an AI based choose your own adventure game created by Latitude. The goal of this project is to build of the gameplay of AI Dungeons and add other features to the game such as inventory, health, and a general throughline.

## The Architecture
The architecture for this project is losely based off of an MVC architecture. A model class keeps track of game state. It updates the game state using a neural network to generate text given the previous game state. The neural network is contained in a separate class to make it easier to alter the text generation without affecting the rest of the game. The other two classes, view and controller, are very simple. The view class prints text to the terminal while the controller prompts the user to input text into the terminal. The reason these classes are kept seperate despite their simplicity is that they will most likely represent two different components in the final version of the game which will hopefully be hosted on the website. A simple class structure is defined bellow.

```text
└── Adventure                   [Main class]
    └── Model                   [Keeps track of game state]
        └── Neural Network      [Text generator]
    └── View                    [Displays game to user]
    └── Controller              [Gather input from user]
```

**Questions**  
- Is there a way to run python code on a website?
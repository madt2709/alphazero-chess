class NeuralNet():
    """
    Provides a base class to create neural nets.
    """

    def __init__(self, game):
        pass

    def train(self, examples):
        """
        Takes a list of example games and uses them to train the neural network
        """
        pass

    def predict(self, board):
        """
        Input: 
            - board
        Returns: 
            - pi: a policy vectore for the current board. A numpy array of length equal to the number of legal moves in the current board state.
            - v: current valuation of the position. A float in [-1,1].
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) to a folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of neural network from folder/filename
        """
        pass

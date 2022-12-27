import sys

NUMBER_OF_RES_LAYERS = 16
NUM_OF_MCTS_SEARCHES = 5
L2_REGULARIZATION_PARAMETER = 10**(-4)
# learning rate is as follows:
# - for first 400k iterations: lr = 10**(-2)
# - 400k<600k: lr = 10**(-3)
# - >600k: lr = 10**(-4)
LEARNING_RATE_SCHEDULE = {400: 10**(-2), 600: 10**(-3), sys.maxsize: 10**(-4)}
# batch size to use in data loader
BATCH_SIZE = 32
# num of training games to play before training nnet
NUM_OF_TRAINING_GAMES = 3
EXPLORATION_RATE = 1
NUM_OF_BATCHES = 3

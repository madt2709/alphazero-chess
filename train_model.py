import torch

from nnet.chess_net import ChessNet
from nnet.train import train
from mcts import self_play_one_game
from settings import NUM_OF_TRAINING_GAMES


def train_model(nnet, num_of_training_games=NUM_OF_TRAINING_GAMES):
    """
    Function to complete one sequence of the neural net training process.

    The process is as follows: 
        - Play NUM_OF_TRAINING_GAMES with nnet provided. 
        - Train the nnet with the games played
        - Save the model to the path provided

    Inputs:
        - nnet: nnet to train
        - path: to save the model to
        - num_of_training_games
    """
    dataset = []
    for i in range(num_of_training_games):
        data = self_play_one_game(nnet=nnet)
        dataset += data
    train(nnet, 0.9, dataset)

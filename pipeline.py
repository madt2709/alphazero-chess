import torch
import os

from nnet.chess_net import ChessNet
from settings import NUM_OF_TRAINING_CYCLES, NUM_OF_TRAINING_GAMES
from nnet.train import train
from mcts import self_play_one_game


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


def pipeline(num_of_training_cycles=NUM_OF_TRAINING_CYCLES, nnet_params_path=None):
    if not os.path.exists('model_data'):
        os.mkdir('model_data')
    nnet = ChessNet()
    if nnet_params_path is not None:
        nnet.load_state_dict(torch.load(nnet_params_path))
    base_path = "model_data/"
    if torch.cuda.is_available():
        nnet.cuda()
    for i in range(num_of_training_cycles):
        path = f"model_params_after_{i}_cycles.pt"
        train_model(nnet)
        torch.save(nnet.state_dict(), base_path + path)


if __name__ == "__main__":
    pipeline()

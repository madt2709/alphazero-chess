import torch

from nnet.chess_net import ChessNet
from nnet.train import train
from mcts import self_play_one_game
from settings import NUM_OF_TRAINING_GAMES, NUM_OF_BATCHES


def main():
    """
    Function to start entire process
    """
    nnet = ChessNet()
    dataset = []
    for j in range(NUM_OF_BATCHES):
        for i in range(NUM_OF_TRAINING_GAMES):
            data = self_play_one_game(nnet=nnet)
            dataset += data
        train(nnet, 0.9, dataset)
    torch.save(nnet.state_dict(), 'model_data')


if __name__ == '__main__':
    main()

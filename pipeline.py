import torch

from nnet.chess_net import ChessNet
from train_model import train_model
from settings import NUM_OF_TRAINING_CYCLES


def pipeline(num_of_training_cycles=NUM_OF_TRAINING_CYCLES, nnet_params_path=None):
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

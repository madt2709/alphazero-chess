import torch

from nnet.chess_net import ChessNet
from train_model import train_model


def pipeline(nnet_params_path=""):
    nnet = ChessNet()
    nnet.load_state_dict(torch.load(nnet_params_path))
    if torch.cuda.is_available():
        nnet.cuda()
    path = f"model_params_after_{i}_games"
    for i in range(6):
        train_model(nnet, )

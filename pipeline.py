import torch

from nnet.chess_net import ChessNet


def pipeline(nnet_params_path=""):
    nnet = ChessNet()
    nnet.load_state_dict(torch.load(nnet_params_path))
    if torch.cuda.is_available():
        nnet.cuda()

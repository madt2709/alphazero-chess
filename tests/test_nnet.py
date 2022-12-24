import pytest
import chess
import torch
from pytest_lazyfixture import lazy_fixture

from representations.board import encode_board
from nnet.chess_net import ConvBlock, ResBlock, OutBlock, ChessNet


@pytest.fixture
def starting_board_state():
    return torch.from_numpy(encode_board(chess.Board())).float()


@pytest.fixture
def conv_block():
    return ConvBlock()


@pytest.fixture
def random_input_tensor():
    return torch.rand([1, 256, 8, 8])


@pytest.fixture
def res_block():
    return ResBlock()


@pytest.fixture
def out_block():
    return OutBlock()


@pytest.mark.parametrize(
    'state, output_size', [
        (lazy_fixture('starting_board_state'), torch.Size([1, 256, 8, 8]))
    ]
)
def test_conv_block_output_shape(state, output_size, conv_block):
    assert conv_block(state).size() == output_size


@pytest.mark.parametrize(
    'rand_tensor, output_size', [
        (lazy_fixture('random_input_tensor'), torch.Size([1, 256, 8, 8]))
    ]
)
def test_res_block_output_size(rand_tensor, output_size, res_block):
    assert res_block(rand_tensor).size() == output_size


@pytest.mark.parametrize(
    'rand_tensor, output_sizes', [
        (lazy_fixture('random_input_tensor'), [
         torch.Size([1, 8*8*73]), torch.Size([1, 1])])
    ]
)
def test_out_block_output_sizes(rand_tensor, output_sizes, out_block):
    p, v = out_block(rand_tensor)
    assert p.size() == output_sizes[0], "policy size is not right"
    assert v.size() == output_sizes[1], "value size is not right"

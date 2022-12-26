import pytest
import chess
from pytest_lazyfixture import lazy_fixture
import numpy as np
import torch

from mcts import UCTNode, DummyNode, get_next_state
from representations.board import encode_board, decode_board
from nnet.chess_net import ChessNet


@pytest.fixture
def uct_node_of_starting_board():
    return UCTNode(encode_board(chess.Board()), None, DummyNode())


@pytest.fixture
def chess_net():
    return ChessNet()


@pytest.fixture
def starting_board_state():
    return torch.from_numpy(encode_board(chess.Board())).float()


@pytest.fixture
def e2e4_played_board():
    board = chess.Board()
    board.push_san('e2e4')
    return board


@pytest.mark.parametrize(
    'utc_node, output', [
        (lazy_fixture('uct_node_of_starting_board'), np.zeros([8*8*73]))
    ]
)
def test_child_Q(utc_node, output):
    assert utc_node.child_Q().shape == output.shape


@pytest.mark.parametrize(
    'utc_node, output', [
        (lazy_fixture('uct_node_of_starting_board'), np.zeros([8*8*73]))
    ]
)
def test_child_U(utc_node, output):
    assert utc_node.child_U().shape == output.shape


@pytest.mark.parametrize(
    'utc_node, output', [
        (lazy_fixture('uct_node_of_starting_board'), np.zeros([8*8*73]))
    ]
)
def test_best_child(utc_node, output):
    assert True


@pytest.mark.parametrize(
    "state, action_idx, next_state", [
        (lazy_fixture('starting_board_state'), 2417,
         lazy_fixture('e2e4_played_board'))
    ]
)
def test_get_next_state(state, action_idx, next_state):
    assert next_state == decode_board(
        get_next_state(state, action_idx))

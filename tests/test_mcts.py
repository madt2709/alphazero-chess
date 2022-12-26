import pytest
import chess
from pytest_lazyfixture import lazy_fixture
import numpy as np

from mcts import UCTNode, DummyNode, self_play_one_game
from representations.board import encode_board
from nnet.chess_net import ChessNet


@pytest.fixture
def uct_node_of_starting_board():
    return UCTNode(encode_board(chess.Board()), None, DummyNode())


@pytest.fixture
def chess_net():
    return ChessNet()


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


def test_self_play_one_game(chess_net):
    self_play_one_game(chess_net)
    assert True

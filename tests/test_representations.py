import pytest
import chess
import numpy as np
from pytest_lazyfixture import lazy_fixture

from representations.position import encode_position, decode_position
from representations.board import encode_board, decode_board
from representations.moves import encode_move, encode_actions, decode_actions, decode_move

# TODO - add more tests


@pytest.fixture
def starting_board():
    return chess.Board()


@pytest.fixture
def encoded_starting_position():
    return encode_position(chess.Board())

# TO DO improve these tests


@pytest.fixture
def encoded_starting_board():
    return encode_board(chess.Board())


@pytest.fixture
def e2e4():
    return chess.Move(chess.E2, chess.E4)


@pytest.fixture
def g7g8n():
    return chess.Move(chess.G7, chess.G8, chess.KNIGHT)


@pytest.fixture
def g1f3():
    return chess.Move(chess.G1, chess.F3)


@pytest.fixture
def e2d3():
    return chess.Move(chess.E2, chess.D3)


@pytest.fixture
def encoded_e2e4():
    return chess.square_file(chess.E2), chess.square_rank(chess.E2), 8


@pytest.fixture
def encoded_g7g8n():
    return chess.square_file(chess.G7), chess.square_rank(chess.G7), 65


@pytest.fixture
def encoded_g1f3():
    return chess.square_file(chess.G1), chess.square_rank(chess.G1), 56


@pytest.fixture
def encoded_e2d3():
    return chess.square_file(chess.E2), chess.square_rank(chess.E2), 0


@pytest.fixture
def starting_board_legal_moves():
    return chess.Board().legal_moves


@pytest.fixture
def encoded_e2e4_played_board():
    board = chess.Board()
    board.push_san('e2e4')
    return encode_board(board)


@pytest.fixture
def encoded_e2e4_played_position():
    board = chess.Board()
    board.push_san('e2e4')
    return encode_position(board)


@pytest.fixture
def e2e4_played_board():
    board = chess.Board()
    board.push_san('e2e4')
    return board


@pytest.fixture
def encoded_starting_board_legal_moves():
    return encode_actions(chess.Board().legal_moves)


@pytest.mark.parametrize(
    "raw,encoded", [
        (lazy_fixture('starting_board'), lazy_fixture('encoded_starting_position'))
    ]
)
def test_encode_position(raw, encoded):
    assert all([a == b] for a, b in zip(encode_position(raw), encoded))


@pytest.mark.parametrize(
    "encoded,raw, colour", [(lazy_fixture('encoded_starting_position'),
                             lazy_fixture('starting_board'), chess.WHITE), (lazy_fixture('encoded_e2e4_played_position'), lazy_fixture('e2e4_played_board'), chess.BLACK)]
)
def test_decode_position(encoded, raw, colour):
    assert decode_position(encoded, colour) == raw


@pytest.mark.parametrize(
    "raw,encoded", [
        (lazy_fixture('starting_board'), lazy_fixture('encoded_starting_board'))
    ]
)
def test_encode_board(encoded, raw):
    assert all([a == b] for a, b in zip(encode_board(raw), encoded))


@pytest.mark.parametrize(
    "encoded, raw", [
        (lazy_fixture('encoded_starting_board'),
         lazy_fixture('starting_board')),
        (lazy_fixture('encoded_e2e4_played_board'),
         lazy_fixture('e2e4_played_board'))
    ]
)
def test_decode_board(encoded, raw):
    assert decode_board(encoded) == raw


@pytest.mark.parametrize(
    "raw, encoded", [
        (lazy_fixture('e2e4'), lazy_fixture('encoded_e2e4')),
        (lazy_fixture('g7g8n'), lazy_fixture('encoded_g7g8n')),
        (lazy_fixture('g1f3'), lazy_fixture('encoded_g1f3')),
        (lazy_fixture('e2d3'), lazy_fixture('encoded_e2d3'))
    ]
)
def test_encode_move(encoded, raw):
    assert all([a == b] for a, b in zip(encode_move(raw), encoded))


@pytest.mark.parametrize(
    "raw, encoded", [
        (lazy_fixture('e2e4'), lazy_fixture('encoded_e2e4')),
        (lazy_fixture('g7g8n'), lazy_fixture('encoded_g7g8n')),
        (lazy_fixture('g1f3'), lazy_fixture('encoded_g1f3')),
        (lazy_fixture('e2d3'), lazy_fixture('encoded_e2d3'))
    ]
)
def test_decode_move(encoded, raw):
    assert raw == decode_move(encoded[0], encoded[1], encoded[2])


@pytest.mark.parametrize(
    "raw, encoded", [
        (lazy_fixture('starting_board_legal_moves'),
         lazy_fixture('encoded_starting_board_legal_moves'))
    ]
)
def test_encode_actions(raw, encoded):
    assert all([a == b] for a, b in zip(encode_actions(raw), encoded))


@pytest.mark.parametrize(
    "raw, encoded", [
        (lazy_fixture('starting_board_legal_moves'),
         lazy_fixture('encoded_starting_board_legal_moves'))
    ]
)
def test_decode_actions(raw, encoded):
    assert set([move.uci() for move in decode_actions(encoded)]) == set([
        move.uci() for move in raw])

# @pytest.mark.parametrize(
#     "start_position, end_position, move", [
#         (lazy_fixture('encoded_starting_board'),
#          lazy_fixture('encoded_e2e4_played_board'), [4, 1, 4, 3, 1])
#     ]
# )
# def test_find_difference_between_encoded_positions(start_position, end_position, move):
#     assert all([a == b] for a, b in zip(
#         find_difference_between_encoded_positions(start_position, end_position), move))

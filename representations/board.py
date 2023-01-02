# this file contains the encoder + decoder for a board

import chess
import numpy as np
from representations.position import encode_position, decode_position


def encode_board(board: chess.Board()):
    """
    A function to encode a board into an object the nnet can process

    Input: 
        - chess.Board()

    Output: 
        - 8 x 8 x ((6 + 6 + 2) x 8 + 1 x 1 x 2 x 2 x 1) array representing in order: 
        chessboard x (P1 piece x P2 piece x Repetitions x 8-step history + colour x total move count x P1 castling x P2 castling x no progress count)
    The first 14 columns will be the board position at time t, the next 14 columns the board position at time t-1, etc..
    Once the 8 step history is complete, the extra variables will be added.
    """
    encoded = np.zeros([8, 8, 119]).astype(float)

    # encode historic boards. Make copy of board to safely remove from move stack
    ply_count = board.ply()
    board_copy = board.copy()
    for i in range(8):
        if ply_count - i >= 0:
            encoded_position = encode_position(board_copy)
            for (j, k, l), value in np.ndenumerate(encoded_position):
                encoded[j][k][l+14*i] = value
            if not ply_count - i <= 1:
                board_copy.pop()

    # add colour to encoding. Note 112 = 14*8
    colour = 1 if board.turn else 0
    encoded[:, :, 112] = colour

    # add total move count
    encoded[:, :, 113] = board.ply()

    # add castling rights
    encoded[:, :, 114] = 1 if board.has_kingside_castling_rights(
        board.turn) else 0
    encoded[:, :, 115] = 1 if board.has_queenside_castling_rights(
        board.turn) else 0
    encoded[:, :, 116] = 1 if board.has_kingside_castling_rights(
        not board.turn) else 0
    encoded[:, :, 117] = 1 if board.has_queenside_castling_rights(
        not board.turn) else 0

    # add no progress count
    encoded[:, :, 118] = board.halfmove_clock

    return encoded


def decode_board(encoded_board):
    """
    A function to decode an encoded board into a chess.Board() with move stack

    Input: 
        - 8 x 8 x ((6 + 6 + 2) x 8 + 1 x 1 x 2 x 2 x 1) array representing in order: 
        chessboard x (P1 piece x P2 piece x Repetitions x 8-step history + colour x total move count x P1 castling x P2 castling x no progress count)

    Output: 
        - chess.Board() without move stack. Getting the move stack is really computer intensive and not actually necessary to the implementation.
    """
    # work out colour. Note we need the opposite colour to most recent position since at oldest step history, the color is inverse.
    colour = True if encoded_board[0][0][112] == 1 else False

    # create board with oldest position. Will make moves in order
    board = decode_position(encoded_board[:, :, :14], colour)

    # retrieve castling rights
    fen = ''
    if encoded_board[0][0][114] == 1:
        fen += 'K' if board.turn else 'k'
    if encoded_board[0][0][115] == 1:
        fen += 'Q' if board.turn else 'q'
    if encoded_board[0][0][116] == 1:
        fen += 'k' if board. turn else 'K'
    if encoded_board[0][0][117] == 1:
        fen += 'q' if board.turn else 'Q'
    if not board.turn:
        fen = fen[::-1]
    board.set_castling_fen(fen)

    return board

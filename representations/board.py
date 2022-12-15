# this file contains the encoder + decoder for a board

import chess
import numpy as np


def encode_board(board: chess.Board()):
    """
    A function to encode a board into an object the nnet can process
    Input: 
        - chess.Board()
    Output: 
        - 64 x ((6 + 6 + 2) x 8 + 1 x 1 x 2 x 2 x 1) array representing in order: 
        chessboard x (P1 piece x P2 piece x Repetitions x 8-step history + colour x total move count x P1 castling x P2 castling x no progress count)
    The first 14 columns will be the board position at time t, the next 14 columns the board position at time t-1, etc..
    Once the 8 step history is complete, the extra variables will be added.
    """
    encoded = np.zeros([64, 119])

    # encode historic boards. Make copy of board to safely remove from move stack
    ply_count = board.ply()
    board_copy = board.copy()
    for i in range(8):
        if ply_count - i >= 0:
            encoded_pieces = encode_pieces(board_copy)
            for index, value in np.ndenumerate(encoded_pieces):
                encoded[index[0]][index[1]+8*i] = value
            board.pop()

    # add colour to encoding. Note 112 = 14*8
    if board.turn:
        colour = 1
    else:
        colour = 0
    encoded[:][112] = colour

    # add total move count
    encoded[:][113] = board.ply()

    # add castling rights
    encoded[:][114] = 1 if board.has_kingside_castling_rights(
        board.turn) else 0
    encoded[:][115] = 1 if board.has_queenside_castling_rights(
        board.turn) else 0
    encoded[:][116] = 1 if board.has_kingside_castling_rights(
        not board.turn) else 0
    encoded[:][117] = 1 if board.has_queenside_castling_rights(
        not board.turn) else 0

    # add no progress count
    encoded[:][118] = board.halfmove_clock

    return encoded


def encode_pieces(board: chess.Board()):
    """
    A fucntion to encode the pieces on a chess.Board()
    Input: 
        -chess.Board()
    Output: 
        - 64 x (6 + 6 + 2) array representing: 
        chessboard x (P1 piece + P2 piece + Repetitions)
    """
    encoded = np.zeroes([64, 14])

    # encode pieces
    for piece in chess.PIECE_TYPES:
        p1_squares = board.pieces(piece, board.turn)
        p2_squares = board.pieces(piece, not board.turn)
        for square in p1_squares:
            encoded[square][piece] = 1
        for square in p2_squares:
            encoded[square][piece+6] = 1

    # encode repetitions
    for i in range(3):
        if board.is_repetition(i+1):
            encoded[:][12] = i+1
            encoded[:][13] = i

    return encoded

# this file contains the encoder + decoder for a position

import chess
import numpy as np


def encode_position(board: chess.Board()):
    """
    A fucntion to encode the pieces on a chess.Board()
    Input: 
        -chess.Board()
    Output: 
        - 64 x (6 + 6 + 2) array representing: 
        chessboard x (P1 piece + P2 piece + Repetitions)
    """
    encoded = np.zeros([64, 14]).astype(int)

    # encode pieces
    for piece in chess.PIECE_TYPES:
        p1_squares = board.pieces(piece, board.turn)
        p2_squares = board.pieces(piece, not board.turn)
        for square in p1_squares:
            encoded[square][piece-1] = 1
        for square in p2_squares:
            encoded[square][piece+5] = 1

    # encode repetitions
    for i in range(3):
        if board.is_repetition(i+1):
            encoded[:, 12] = i+1
            encoded[:, 13] = i

    return encoded


def decode_position(encoded_position, colour):
    """
    A function to decode an encoded position into a chess.Board() position
    Input: 
        - 64 x (6 + 6 + 2) array representing: 
        chessboard x (P1 piece + P2 piece + Repetitions)
    Output: 
        -chess.Board() with no move stack.

    TO DO: set up repetitions.
    """
    pieces_dict = {}
    board = chess.Board()
    for i in range(64):
        for j in range(6):
            if encoded_position[i][j] == 1:
                pieces_dict[i] = chess.Piece(j+1, colour)
            if encoded_position[i][j+6] == 1:
                pieces_dict[i] = chess.Piece(j+1, not colour)

    board.set_piece_map(pieces_dict)
    return board

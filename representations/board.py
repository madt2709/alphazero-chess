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
    encoded = np.zeros([64, 119]).astype(int)

    # encode historic boards. Make copy of board to safely remove from move stack
    ply_count = board.ply()
    board_copy = board.copy()
    for i in range(8):
        if ply_count - i >= 0:
            encoded_position = encode_position(board_copy)
            for index, value in np.ndenumerate(encoded_position):
                encoded[index[0]][index[1]+8*i] = value
            board.pop()

    # add colour to encoding. Note 112 = 14*8
    if board.turn:
        colour = 1
    else:
        colour = 0
    encoded[:, 112] = colour

    # add total move count
    encoded[:, 113] = board.ply()

    # add castling rights
    encoded[:, 114] = 1 if board.has_kingside_castling_rights(
        board.turn) else 0
    encoded[:, 115] = 1 if board.has_queenside_castling_rights(
        board.turn) else 0
    encoded[:, 116] = 1 if board.has_kingside_castling_rights(
        not board.turn) else 0
    encoded[:, 117] = 1 if board.has_queenside_castling_rights(
        not board.turn) else 0

    # add no progress count
    encoded[:, 118] = board.halfmove_clock

    return encoded


def encode_position(board: chess.Board()):
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


def decode_board(encoded_board):
    """
    A function to decode an encoded board into a chess.Board() with move stack
    Input: 
        - 64 x ((6 + 6 + 2) x 8 + 1 x 1 x 2 x 2 x 1) array representing in order: 
        chessboard x (P1 piece x P2 piece x Repetitions x 8-step history + colour x total move count x P1 castling x P2 castling x no progress count)
    Output: 
        - chess.Board() with move stack
    """
    # check which is the oldest non-zero board.
    i = 7
    while not encoded_board[:, i*14:(i+1)*14].any():
        i -= 1

    # create board with oldest position. Will make moves in order
    board = decode_position(encoded_board[:, i*14:(i+1)*14])

    # work out moves to get to most recent position
    while i >= 1:
        start_square, end_square, final_piece = find_difference_between_encoded_positions(
            encoded_board[:, i*14:(i+1)*14], encoded_board[:, (i-1)*14:i*14])
        board.push(chess.Move(start_square, end_square, final_piece))
        i -= 1

    return board


def find_difference_between_encoded_positions(encoded_start_position, encoded_end_position):
    """
    A function which takes a start encoded position and an end encoded position and returns the coordinates of the move diff
    """
    difference = encoded_end_position - encoded_start_position

    # looking for plane in difference where there is 1. note pieces are only added to a position in the future so the only way to get a 1 is by a piece moving there.
    # initialise with negative numbers to make check easy. Track final piece in case of promotion
    start_square, end_square, final_piece = -1, -1, -1
    while True:
        for j in range(12):
            for i in range(64):
                if difference[i][j] == 1:
                    end_square, piece = i, j
                elif difference[i][j] == -1:
                    start_square = i
            # check if we have the relevant data
            if start_square >= 0 and end_square >= 0 and final_piece >= 0:
                break

    # check we are not in edge case of rook making castling move
    if final_piece == 4 | 10 and ((start_square, end_square) == (0, 3) or (7, 4) or (56, 59) or (63, 60)):
        # check if king has also moved. If yes, then update the start square, end square and final piece
        for i in range(64):
            if difference[i][final_piece + 2] == 1:
                end_square, piece = i, final_piece + 2
            elif difference[i][final_piece + 2] == -1:
                start_square = i

    return start_square, end_square, final_piece

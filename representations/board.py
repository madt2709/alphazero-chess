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
        - chess.Board() with move stack
    """
    # work out colour. Note we need the opposite colour to most recent position since at oldest step history, the color is inverse.
    colour = True if encoded_board[0][0][112] == 1 else False

    # create board with oldest position. Will make moves in order
    board = decode_position(encoded_board[:, :, :14], colour)

    # this is really CPU intensive will skip for now.
    # # work out moves to get to most recent position
    # while i >= 1:
    #     start_file, start_rank, end_file, end_rank, final_piece = find_difference_between_encoded_positions(
    #         encoded_board[:, :, i*14:(i+1)*14], encoded_board[:, :, (i-1)*14:i*14])
    #     board.push(chess.Move(chess.square(start_file, start_rank),
    #                chess.square(end_file, end_rank), final_piece))
    #     i -= 1

    return board


# def find_difference_between_encoded_positions(encoded_start_position, encoded_end_position):
#     """
#     A function which takes a start encoded position and an end encoded position and returns the coordinates of the move diff
#     """
#     difference = encoded_end_position - encoded_start_position
#     # looking for plane in difference where there is 1. note pieces are only added to a position in the future so the only way to get a 1 is by a piece moving there.
#     # initialise with negative numbers to make check easy. Track final piece in case of promotion
#     start_square, end_square, final_piece = None, None, None
#     while not start_square or not end_square or not final_piece:
#         for (i, j, k), value in np.ndenumerate(difference[:, :, :12]):
#             if value == 1:
#                 end_file, end_rank, final_piece = i, j, k
#             if value == -1:
#                 start_file, start_rank = i, j

#     # check we are not in edge case of rook making castling move
#     if final_piece == 4 | 10 and (((start_file, start_rank), (end_file, end_rank))) == ((0, 0), (3, 0)) or ((7, 0), (4, 0)) or ((0, 7), (3, 7)) or ((7, 7), (4, 7)):
#         # check if king has also moved. If yes, then update the start square, end square and final piece
#         for i in range(8):
#             for j in range(8):
#                 if difference[i][j][final_piece + 2] == 1:
#                     end_file, end_rank, final_piece = i, j, final_piece + 2
#                 elif difference[i][j][final_piece + 2] == -1:
#                     start_file, start_rank = i, j

#     return start_file, start_rank, end_file, end_rank, final_piece

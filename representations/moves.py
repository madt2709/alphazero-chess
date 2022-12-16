# this file contains the encoder/decoder for actions

import chess
import numpy as np


def encode_actions(board: chess.Board()):
    """
    A function to encode the set of legal moves in chess.Board().
    Input: 
        - chess.Board()
    Output:
        - a 64 x 72 array where the 64 represents the piece to "pick up". 72 dimension is broken down into the following: 
        56 represents possible "queen moves" for a piece. This is broken into number of squares [1,..,7] and the relative compass directions ["N", "NE", "E", ... , "NW"]. The first 7 entries will be 
        in ascending order of number of squares in "N" direction and then we will go clockwise through the directions. 
        8 features representing possible night moves going in clockwise order starting with move where horse goes 2 up and 1 to the side. 
        9 features for underpromotion to knight, bishop or rook. Any other promotion will be assumed to be a queen
    """
    encoded = np.zeros([64, 73])

    for move in board.legal_moves:
        start_square, repr = encode_move(move)
        encoded[start_square][repr] == 1

    return encoded


def encode_move(move: chess.Move()):
    """A function to encode a move into one of the following representations: 
        56 represents possible "queen moves" for a piece. This is broken into number of squares [1,..,7] and the relative compass directions ["NW", "N", "NE", ... , "W"]. The first 7 entries will be 
        in ascending order of number of squares in "NW" direction and then we will go clockwise through the directions. 
        8 features representing possible night moves going in clockwise order starting with move where horse goes 2 up and 1 to the left. 
        9 features for underpromotion to knight, bishop or rook. Any other promotion will be assumed to be a queen. Will do knight promotions in order ["NW", "N", "NE"] then bishop and finally rook. 
    Input: 
        - chess.Move()
    Output:
        - start_square: piece starting square
        - number between 1,..,73 according to the representation
    """
    uci = move.uci()
    direction_counter = 0
    counter = 0

    # convert to coordinates
    letter_converter = {"a": 1, "b": 2, "c": 3,
                        "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
    x_start, y_start = letter_converter[uci[0]], int(uci[1])
    x_end, y_end = letter_converter[uci[2]], int(uci[3])

    # work out direction
    dx, dy = x_end - x_start, y_end - y_start
    if dx == - dy and dy > 0:
        # NW
        direction_counter = 0
    elif dx == 0 and dy > 0:
        # N
        direction_counter = 1
    elif dx == dy and dy > 0:
        # NE
        direction_counter = 2
    elif dx > 0 and dy == 0:
        # E
        direction_counter = 3
    elif dx == - dy and dy < 0:
        # SE
        direction_counter = 4
    elif dx == 0 and dy < 0:
        # S
        direction_counter = 5
    elif dx == dy and dy < 0:
        # SW
        direction_counter = 6
    elif dx < 0 and dy == 0:
        # W
        direction_counter = 7
    else:
        # must be a knight move
        counter = 56
        if dx == -1 and dy == 2:
            direction_counter = 0
        elif dx == 1 and dy == 2:
            direction_counter = 1
        elif dx == 2 and dy == 1:
            direction_counter = 2
        elif dx == 2 and dy == -1:
            direction_counter = 3
        elif dx == 1 and dy == -2:
            direction_counter = 4
        elif dx == -1 and dy == -2:
            direction_counter = 5
        elif dx == -2 and dy == -1:
            direction_counter = 6
        elif dx == -2 and dy == 1:
            direction_counter = 7
        # identified type of move + direction so can return at this point
        return move.from_square, counter + direction_counter

    # check if move is underpromotion
    if uci[-1] == "k":
        counter = 63
        return move.from_square, counter + direction_counter
    elif uci[-1] == "b":
        counter = 66
        return move.from_square, counter + direction_counter
    elif uci[-1] == "r":
        counter = 69
        return move.from_square, counter + direction_counter

    # find length of move
    counter = max(abs(dx), abs(dy))

    return move.from_square, direction_counter*7 + counter


def decode_actions(encoded_actions):
    """
    A function to decode actions. 
    Inputs:
        - 64 x 73 array
    Outputs:
        - chess.Board().legal_moves
    """
    legal_moves = []
    direction_converter = np.array([[-1, 1], [0, 1], [1, 1],
                                    [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]])
    knight_converter = np.array([[-1, 2], [1, 2], [2, 1], [2, -1],
                                 [1, -2], [-1, -2], [-2, -1], [-2, 1]])
    underpromotion_converter = np.array([-1, 1], [0, 1], [1, 1])
    for (i, j), value in np.ndenumerate(encoded_actions):
        if value == 1:
            start = np.array([chess.square_file(i), chess.square_rank(i)])
            if j < 56:
                length = j % 7
                direction = 0
                while direction*7 < j:
                    direction += 1
                end = start + direction_converter[direction] * length
                move = chess.Move(i, chess.square(end[0], end[1]), chess.QUEEN)
                legal_moves.append(move)
            elif 56 <= j < 64:
                direction = j % 8
                end = start + knight_converter[direction]
                move = chess.Move(i, chess.square(end[0], end[1]))
                legal_moves.append(move)
            else:
                direction = j - 64 % 3
                piece = 1
                while 64 + 3*piece <= j:
                    piece += 1
                end = start + underpromotion_converter[direction]
                move = chess.Move(i, chess.square(end[0], end[1], piece+1))
                legal_moves.append(move)

    return legal_moves

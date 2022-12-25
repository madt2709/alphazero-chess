# this file contains the encoder/decoder for actions

import chess
import numpy as np


def encode_actions(legal_moves: chess.Board().legal_moves):
    """
    A function to encode the set of legal moves in chess.Board().
    Input: 
        - chess.Board().legal_moves
    Output:
        - a 8 x 8 x 73 array where the 8 x 8 represents the piece to "pick up". 72 dimension is broken down into the following: 
        56 represents possible "queen moves" for a piece. This is broken into number of squares [1,..,7] and the relative compass directions ["NW", "N", "NE", ... , "W"]. The first 7 entries will be 
        in ascending order of number of squares in "N" direction and then we will go clockwise through the directions. 
        8 features representing possible night moves going in clockwise order starting with move where horse goes 2 up and 1 to the side. 
        9 features for underpromotion to knight, bishop or rook. Any other promotion will be assumed to be a queen
        This will however be converted to a 8*8*73 so that it is compatible with neural net output.
    """
    encoded = np.zeros([8, 8, 73]).astype(float)

    for move in legal_moves:
        square_file, square_rank, repr = encode_move(move)
        encoded[square_file][square_rank][repr] = 1

    return encoded.reshape(-1)


def encode_move(move):
    """A function to encode a move into one of the following representations: 
        56 represents possible "queen moves" for a piece. This is broken into number of squares [1,..,7] and the relative compass directions ["NW", "N", "NE", ... , "W"]. The first 7 entries will be 
        in ascending order of number of squares in "NW" direction and then we will go clockwise through the directions. 
        8 features representing possible night moves going in clockwise order starting with move where horse goes 2 up and 1 to the left. 
        9 features for underpromotion to knight, bishop or rook. Any other promotion will be assumed to be a queen. Will do knight promotions in order ["NW", "N", "NE"] then bishop and finally rook. 
    Input: 
        - chess.Move()
    Output:
        - start_file: piece starting file
        - start_rank: piece starting rank
        - number between 0,..,72 according to the representation
    """
    uci = move.uci()
    start_file, start_rank = (chess.square_file(
        move.from_square), chess.square_rank(move.from_square))
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
        return start_file, start_rank, counter + direction_counter

    # check if move is underpromotion
    if uci[-1] == "n":
        counter = 63
        return start_file, start_rank, counter + direction_counter
    elif uci[-1] == "b":
        counter = 66
        return start_file, start_rank, counter + direction_counter
    elif uci[-1] == "r":
        counter = 69
        return start_file, start_rank, counter + direction_counter

    # find length of move
    counter = max(abs(dx), abs(dy)) - 1

    return start_file, start_rank, direction_counter*7 + counter


def decode_actions(encoded_actions):
    """
    A function to decode actions. 
    Inputs:
        - 8 x 8 x 73 array
    Outputs:
        - chess.Board().legal_moves
    """
    encoded_actions = encoded_actions.reshape(8, 8, 73)
    legal_moves = []
    for (i, j, k), value in np.ndenumerate(encoded_actions):
        if value == 1:
            move = decode_move(i, j, k)
            legal_moves.append(move)

    return legal_moves


def decode_move(start_file, start_rank, move_type):
    direction_converter = np.array([[-1, 1], [0, 1], [1, 1],
                                    [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]])
    knight_converter = np.array([[-1, 2], [1, 2], [2, 1], [2, -1],
                                 [1, -2], [-1, -2], [-2, -1], [-2, 1]])
    underpromotion_converter = np.array([[-1, 1], [0, 1], [1, 1]])
    if move_type < 56:
        # note can't calculate length straight away since 7 is a value here.
        co_ord = move_type % 7
        length = co_ord + 1
        direction = 0
        while (direction+1)*7 <= move_type:
            direction += 1
        end = [start_file, start_rank] + \
            direction_converter[direction] * length
        move = chess.Move(chess.square(start_file, start_rank), chess.square(
            end[0], end[1]))
    elif 56 <= move_type < 64:
        direction = move_type % 8
        end = [start_file, start_rank] + knight_converter[direction]
        move = chess.Move(chess.square(start_file, start_rank),
                          chess.square(end[0], end[1]))
    else:
        direction = (move_type - 64) % 3
        piece = 1
        while 64 + 3*piece <= move_type:
            piece += 1
        end = [start_file, start_rank] + underpromotion_converter[direction]
        move = chess.Move(chess.square(start_file, start_rank),
                          chess.square(end[0], end[1]), piece+1)
    return move

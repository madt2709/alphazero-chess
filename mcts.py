import chess
import numpy as np
import math
import collections
import torch
import random

from representations.moves import decode_move, encode_actions
from representations.board import encode_board, decode_board
from settings import NUM_OF_MCTS_SEARCHES, EXPLORATION_RATE
import mcts


class UCTNode():
    """
    A class to create an upper confidence node in MCTS
    """

    def __init__(self, state, move, parent=None):
        """
        Inputs: 
            - state: state of the game
            - move: encoded last move played. (ie (starting square, move type))
            - parent: parent node if it exists
        """
        self.s = state
        self.move = move
        self.parent = parent
        self.children = {}  # key = idx_of_action required to reach child node, value = child node
        self.child_priors = np.zeros([4672]).astype(
            float)  # store previous prediction of nnet
        self.child_number_of_visits = np.zeros([4672]).astype(float)
        self.child_total_value = np.zeros([4672]).astype(float)

    @property
    def number_of_visits(self):
        return self.parent.child_number_of_visits[self.move]

    @number_of_visits.setter
    def number_of_visits(self, value):
        self.parent.child_number_of_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    @property
    def legal_actions(self):
        # store the legal actions of given state
        board = decode_board(self.s.numpy())
        legal_actions = encode_actions(board.legal_moves, board.turn)
        return np.where(legal_actions == 1)[0]

    def child_Q(self):
        return self.child_total_value / (self.child_number_of_visits + 1)

    def child_U(self):
        if type(self.parent) is mcts.DummyNode:
            return np.zeros([4672]).astype(float)
        else:
            return math.sqrt(self.number_of_visits) * (
                abs(self.child_priors) / (1 + self.child_number_of_visits))

    def add_dirichlet_noise(self):
        noise = np.random.default_rng().dirichlet(
            np.zeros([len(self.legal_actions)], dtype=np.float32)+0.3)
        for i, value in enumerate(self.legal_actions):
            self.child_priors[value] = 0.75 * \
                self.child_priors[value] + 0.25*noise[i]

    def best_child(self):
        func_to_max = self.child_U() + self.child_Q()
        max_value = -10000
        max_idx = -1
        eq_values = []
        for idx in self.legal_actions:
            if func_to_max[idx] > max_value:
                max_idx = idx
                max_value = func_to_max[idx]
                eq_values = []
            elif func_to_max[idx] == max_value:
                eq_values.append(idx)
        if eq_values != []:
            eq_values.append(max_idx)
            max_idx = random.choice(eq_values)
        return max_idx

    def check_if_child_node_exists(self, action_idx):
        return action_idx in self.children.keys()

    def search(self, nnet):
        """
        Method to perform MCTS search on a given node. 
        Inputs:
            -nnet: neural net used to evaluate position
        Outputs:
            - total_value: value determined by MCTS algo
        Note we return the negative value becaue we expect the next search to be from perspective of other player.
        """
        # check if game has ended
        board = decode_board(self.s.numpy())
        outcome = board.outcome()
        if outcome:
            if outcome.winner == board.turn:
                self.total_value = 1
            elif outcome.winner is None:
                self.total_value = 0
            else:
                self.total_value = -1
            return -self.total_value

        # find action a which maximises U
        best_action = self.best_child()

        # check if child exists
        if not self.check_if_child_node_exists(best_action):
            next_s = get_next_state(self.s, best_action)
            # create child
            child = UCTNode(next_s, best_action, self)
            # predict
            p_s, self.child_total_value[best_action] = nnet(next_s)
            # assign p_s to child priors removing any values for illegal moves
            p_s = p_s.detach().cpu().numpy().reshape(-1)
            for idx in range(len(p_s)):
                if idx not in child.legal_actions:
                    p_s[idx] = 0.0
            child.child_priors = p_s
            # add dirichlet noise to root node
            if child.parent.parent == None:
                child.add_dirichlet_noise()
            # assign to child
            self.children[best_action] = child
            self.child_number_of_visits[best_action] += 1
            return -self.child_total_value[best_action]
        else:
            self.child_number_of_visits[best_action] += 1
            return self.children[best_action].search(nnet)

    def backpropogate(self, value):
        current = self
        # use this to track the colour of the player's value being updated.
        counter = True
        while current.parent is not None:
            current.child_number_of_visits[self.move] += 1
            if counter:
                current.total_value += 1*value
            else:
                current.total_value += -1*value
            current = current.parent
            counter = not counter


def get_next_state(s, action_idx):
    i, j, k = np.unravel_index(action_idx, [8, 8, 73])
    board = decode_board(s.numpy())
    move = decode_move(
        i, j, k, board.turn)
    # make move
    if board.piece_at(move.from_square) == 1 and (board.turn, chess.square_rank(move.from_square)) in [(True, 6), (False, 1)] and move.promotion == None:
        move.promotion = chess.QUEEN
    board.push(move)
    # encode next board
    next_state = encode_board(board)
    return torch.from_numpy(next_state)


class DummyNode():
    """
    A dummy class used when initalising an MCTS iteration
    """

    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_of_visits = collections.defaultdict(float)


def complete_one_mcts(num_of_searches, nnet, starting_position=chess.Board()):
    root = UCTNode(torch.from_numpy(encode_board(starting_position)).float(),
                   move=None, parent=DummyNode())
    for i in range(num_of_searches):
        value = root.search(nnet)
        root.backpropogate(value)
    return np.argmax(root.child_number_of_visits), root


def get_policy(node):
    """
    Function to get policy of a node. Should really only be called on root of searches
    """
    policy = np.zeros(8*8*73).astype(int)
    for idx in np.where(node.child_number_of_visits != 0)[0]:
        policy[idx] = node.child_number_of_visits[idx] / \
            node.child_number_of_visits.sum()
    return policy


def self_play_one_game(nnet, num_of_search_iters=NUM_OF_MCTS_SEARCHES, starting_position=chess.Board()):
    """
    A function to play 1 training game.

    Inputs:
        - num_of_search_iters: this is used to know how many iterations the MCTS algo should perform before picking the best move
        - nnet: neural net used to evaluate a position
        - starting_position: the starting position of the training games

    Outputs: 
        - a list where each entry is [s,p,v] for each of the states, policy and values encountered in the training game. 
        Note values are updated to match game outcome.
    eg [[s_1,p_1,v_1],...]
    """
    dataset = []  # to add [s,p] encountered
    dataset_v = []  # to add [s,p,v] once game is over
    board = starting_position.copy()
    while not board.outcome():
        print(board)
        print("\n")
        best_move, root = complete_one_mcts(num_of_search_iters, nnet, board)
        bm_start_file, bm_start_rank, bm_move_type = np.unravel_index(best_move, [
                                                                      8, 8, 73])
        policy = get_policy(root)
        dataset.append([root.s, policy])
        move = decode_move(bm_start_file,
                           bm_start_rank, bm_move_type, board.turn)
        if board.piece_at(move.from_square) == 1 and (board.turn, chess.square_rank(move.from_square)) in [(True, 6), (False, 1)] and move.promotion == None:
            move.promotion = chess.QUEEN
        board.push(decode_move(bm_start_file,
                   bm_start_rank, bm_move_type, board.turn))
    print(board)
    if board.outcome().winner == True:  # white win
        v = 1
        print("White win")
    elif board.outcome().winner == False:
        v = -1
        print("Black win")
    else:
        v = 0
        print("Draw")
    for idx, data in enumerate(dataset):
        s, p = data
        dataset_v.append([s, p, v])
    print(f"Game complete")
    return dataset_v

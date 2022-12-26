import chess
import numpy as np
import math
import collections
import torch

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
        self.child_number_of_visits = np.zeros([4672]).astype(int)
        self.child_total_value = np.zeros([4672]).astype(int)

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
        board = decode_board(self.s)
        return encode_actions(board.legal_moves)

    def child_Q(self):
        return self.child_total_value / (self.child_number_of_visits + 1)

    def child_U(self):
        if type(self.parent) is mcts.DummyNode:
            return np.zeros([4672]).astype(int)
        else:
            return EXPLORATION_RATE*get_policy(self.parent)*math.sqrt(sum([sum(i) for i in self.child_number_of_visits]))/(1+self.child_number_of_visits)

    def best_child(self):
        func_to_max = self.child_U() + self.child_Q()
        max_value = -10000
        max_idx = None
        for i, v in enumerate(self.legal_actions):
            if v == 1:
                if max_value < func_to_max[i]:
                    max_value = func_to_max[i]
                    max_idx = i
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
        board = decode_board(self.s)
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
            self.children[best_action] = UCTNode(next_s, best_action, self)
            self.child_number_of_visits[best_action] += 1
            # predict
            p_s, self.child_total_value[best_action] = nnet(next_s)
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
    unravel_idx = np.unravel_index(action_idx, [8, 8, 73])
    move = decode_move(unravel_idx[0], unravel_idx[1], unravel_idx[2])
    board = decode_board(s.numpy())
    # make move
    board.push(move)
    print(move)
    print(board)
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
        print(f"{i} search complete")
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
        - num_of-search_iters: this is used to know how many iterations the MCTS algo should perform before picking the best move
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
        best_move, root = complete_one_mcts(num_of_search_iters, nnet, board)
        policy = get_policy(root)
        dataset.append([root.s, policy])
        board.push(decode_move(best_move))
    if board.outcome().winner == True:  # white win
        v = 1
    elif board.outcome().winner == False:
        v = -1
    else:
        v = 0
    for idx, data in enumerate(dataset):
        s, p = data
        dataset.append([s, p, v])
    return dataset_v

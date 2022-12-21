import chess
import numpy as np
import math
import collections

from representations.moves import decode_move, encode_actions
from representations.board import encode_board, decode_board
from settings import NUM_OF_MCTS_SEARCHES


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
        self.child_number_of_visits = np.zeros([64, 73]).astype(int)
        self.child_total_value = np.zeros([64, 73]).astype(int)
        self.legal_action_idxs = []  # store the idxs of legal actions of given state
        # TO DO add the legal action idxs properly

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

    def child_Q(self):
        return self.child_total_value / (self.child_number_of_visits + 1)

    def child_U(self, exploration_rate):
        return exploration_rate*self.policy*math.sqrt(sum([sum(i) for i in self.child_number_of_visits]))/(1+self.child_number_of_visits)

    def best_child(self):
        if self.action_idxs != []:
            func_to_max = self.child_U + self.child_Q
            max_value = -10000
            max_idx = (-1, -1)
            for i in self.action_idxs:
                if max_value < func_to_max[i]:
                    max_value = func_to_max[i]
                    max_idx = i
        return i

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
        outcome = board.outcome
        if outcome:
            if outcome.winner == board.turn:
                self.total_value = 1
            elif outcome.winner is None:
                self.total_value = 0
            else:
                self.total_value = -1
            return -self.total_value, self

        # find action a which maximises U
        best_action = self.best_child()

        # check if child exists
        if not self.check_if_child_node_exists():
            next_s = get_next_state(self.s, best_action)
            # create child
            self.children[best_action] = UCTNode(next_s, best_action, self)
            self.child_number_of_visits[best_action] += 1
            # predict
            p_s, self.child_total_value[best_action] = nnet.predict(next_s)
            return -self.child_total_value[best_action], self.children[best_action]
        else:
            child_v = self.children[best_action].search()
            self.child_number_of_visits[best_action] += 1
            return self.children[best_action].search()

    def backpropogate(self, value):
        current = self
        # use this to track the colour of the player's value being updated.
        counter = True
        while current is not None:
            current.parent.child_number_of_visits[self.move] += 1
            if counter:
                current.parent.total_value[self.move] += (1*value)
            else:
                current.parent.total_value[self.move] += (-1*value)
            current = current.parent
            counter = not counter


def get_next_state(s, action_idx):
    move = decode_move(action_idx[0], action_idx[1])
    board = decode_board(s)
    # make move
    board.push(move)
    # encode next board
    next_state = encode_board(board)
    return next_state


class DummyNode():
    """
    A dummy class used when initalising an MCTS iteration
    """

    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_of_visits = collections.defaultdict(float)


def complete_one_mcts(num_of_searches, nnet, starting_position=chess.Board()):
    root = UCTNode(encode_board(starting_position),
                   move=None, parent=DummyNode())
    for i in range(num_of_searches):
        value = root.search(nnet)
        root.backpropogate(value)
    return np.argmax(root.child_number_of_visits), root


def get_policy(node):
    """
    Function to get policy of a node. Should really only be called on root of searches
    """
    policy = np.zeros(64, 73)
    for idx, value in np.ndenumerate(node.child_number_of_visits):
        policy[idx] = value / sum([sum(i)
                                  for i in node.child_number_of_visits])
    return policy


def self_play_one_game(num_of_search_iters, nnet, starting_position=chess.Board()):
    """
    A function to play 1 training game.
    Inputs:
        - num_of-search_iters: this is used to know how many iterations the MCTS algo should perform before picking the best move
        - nnet: neural net used to evaluate a position
        - starting_position: the starting position of the training games
    Outputs: 
        - a list where each entry is [s,p,v] for each of the states, policy and values encountered in the training game. 
    eg [[s_1,p_1,v_1],...]
    """
    dataset = []  # to add [s,p, v] encountered
    board = starting_position.copy()
    while not board.outcome:
        best_move, root = complete_one_mcts(NUM_OF_MCTS_SEARCHES, nnet, board)
        policy = get_policy(root)
        dataset.append([root.s, policy, root.total_value])
        board.push(decode_move(best_move))
    return dataset

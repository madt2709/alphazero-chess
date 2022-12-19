import chess
import numpy as np
import math

from representations.moves import decode_move, encode_actions
from representations.board import encode_board, decode_board


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
        self.children = {}
        self.child_number_of_visits = np.zeros([64, 73]).astype(int)
        self.child_total_value = np.zeros([64, 73]).astype(int)
        self.action_idxs = []  # store the idxs of legal actions of given state
        # TO DO set policy to current neural net best estimation

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
        return (self.child_total_value * self.child_number_of_visits + self.total_value) / (self.child_number_of_visits + 1)

    def child_U(self, exploration_rate):
        return self.child_Q + exploration_rate*self.policy*math.sqrt(sum([sum(i) for i in self.child_number_of_visits]))/(1+self.child_number_of_visits)

    def best_child(self):
        if self.action_idxs != []:
            child_U = self.child_U
            max_value = -10000
            max_idx = (-1, -1)
            for i in self.action_idxs:
                if max_value < child_U[i]:
                    max_value = child_U[i]
                    max_idx = i
        return i

    def check_if_child_node_exists(self, state):
        return state in self.children.keys()

    def search(self, nnet):
        """
        Method to perform MCTS search on a given node. Note we return the negative value becaue we expect the next search to be from perspective of other player.
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
            return -self.total_value

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
            return -self.child_total_value[best_action]
        else:
            child_v = self.children[best_action].search()
            self.child_number_of_visits[best_action] += 1
            return self.children[best_action].search()


def get_next_state(s, action_idx):
    move = decode_move(action_idx[0], action_idx[1])
    board = decode_board(s)
    # make move
    board.push(move)
    # encode next board
    next_state = encode_board(board)
    return next_state

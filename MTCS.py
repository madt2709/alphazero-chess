import chess


class MCTS():
    """
    This class handles the MCTS tree
    """

    def __init__(self, game, nnet, metatdata):
        self.game = game
        self.nnet = nnet
        self.metadata = metatdata
        self.Q = {}  # expected reward for taking an action in state s
        self.P = {}  # initial estimate of taking an action from a state s acccording to the policy returned by nnet
        self.N = {}  # number of times an action has been taken from state s

    def search(self, board: chess.Board()):
        """
        Performs one standard iteration of MCTS. 

        Will implement funciton as described here: https://web.stanford.edu/~surag/posts/alphazero.html
        """
        # check for checkmate, stalemate, insufficient material or can claim draw
        if board.is_checkmate():
            return -1

        if board.is_stalemate():
            return 0

        if board.is_insufficient_material():
            return 0

        if board.can_claim_draw():
            return 0

        if board.__str___() not in self.N.keys():
            self.N[board.__str__()] = 0
            self.P[board.__str__()], v = self.nnet.predict(board)
            return -v

        max_u, best_a = -float("inf"), chess.Move(chess.A1, chess.A2)

        # pick action with highest upper confidence bound

import copy
from typing import Optional, TypedDict

from exceptions import AgentException
from connect4 import Connect4


class Node(TypedDict):
    state: Connect4
    value: list['Node'] | float


class MinMaxAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4: Connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.get_move(copy.deepcopy(connect4))

    @staticmethod
    def check_winner(board, piece):
        # Check rows
        for row in range(6):
            for col in range(4):
                if all(board[row][col + i] == piece for i in range(4)):
                    return True
        # Check columns
        for col in range(7):
            for row in range(3):
                if all(board[row + i][col] == piece for i in range(4)):
                    return True
        # Check diagonals
        for row in range(3):
            for col in range(4):
                if all(board[row + i][col + i] == piece for i in range(4)):
                    return True
                if all(board[row + i][col + 3 - i] == piece for i in range(4)):
                    return True
        return False

    @staticmethod
    def count_pattern(board, piece, length):
        count = 0
        for row in range(6):
            for col in range(7 - length + 1):
                if all(board[row][col + i] == piecedd for i in range(length)):
                    count += 1
        for col in range(7):
            for row in range(6 - length + 1):
                if all(board[row + i][col] == piece for i in range(length)):
                    count += 1
        for row in range(6 - length + 1):
            for col in range(7 - length + 1):
                if all(board[row + i][col + i] == piece for i in range(length)):
                    count += 1
                if all(board[row + i][col + length - 1 - i] == piece for i in range(length)):
                    count += 1
        return count

    @staticmethod
    def evaluate_state_float(board: list, player_piece, opponent_piece) -> float:
        """
        Evaluate the current state of the game board.

        Parameters:
        - board: 2D array representing the game board
        - player_piece: The piece representing the player
        - opponent_piece: The piece representing the opponent

        Returns:
        - score: A numerical score indicating the likelihood of winning for the player
        """

        # Define constants for evaluation
        WIN_SCORE = 100000
        THREE_SCORE = 100
        TWO_SCORE = 10
        ONE_SCORE = 1

        # Check for winning positions
        if MinMaxAgent.check_winner(board, player_piece):
            return WIN_SCORE / WIN_SCORE
        if MinMaxAgent.check_winner(board, opponent_piece):
            return -WIN_SCORE / WIN_SCORE

        # Initialize score
        score = 0

        # # Check for potential winning positions for the player
        # score += MinMaxAgent.count_pattern(board, player_piece, 4) * WIN_SCORE
        # score += MinMaxAgent.count_pattern(board, player_piece, 3) * THREE_SCORE
        # score += MinMaxAgent.count_pattern(board, player_piece, 2) * TWO_SCORE
        # score += MinMaxAgent.count_pattern(board, player_piece, 1) * ONE_SCORE

        # Check for potential winning positions for the opponent
        score -= MinMaxAgent.count_pattern(board, opponent_piece, 4) * WIN_SCORE
        score -= MinMaxAgent.count_pattern(board, opponent_piece, 3) * THREE_SCORE
        score -= MinMaxAgent.count_pattern(board, opponent_piece, 2) * TWO_SCORE
        score -= MinMaxAgent.count_pattern(board, opponent_piece, 1) * ONE_SCORE

        return score / WIN_SCORE

    def calculate_state(self, state: Connect4) -> float:
        other = 'x'
        if other == self.my_token:
            other = 'o'
        return MinMaxAgent.evaluate_state_float(state.board, self.my_token, other)

    @staticmethod
    def eval_states(states: list[Node]) -> float:
        values: list[float] = []
        for state in states:
            if type(state['value']) is not float and type(state['value']) is not int:
                res = MinMaxAgent.eval_states(state['value'])
            else:
                res = state['value']
            values.append(res)
        return min(values)

    def do_move(self, connect4: Connect4, move: int) -> tuple[Connect4, Optional[int]]:
        """Returns copied connect4 game instance with performed move."""
        new_connect4 = copy.deepcopy(connect4)
        new_connect4.drop_token(move)
        if new_connect4.game_over:
            if new_connect4.wins == self.my_token:
                return new_connect4, 1
            elif new_connect4.wins != self.my_token:
                return new_connect4, -1
            else:
                return new_connect4, 0
        return new_connect4, None

    def generate_states(self, connect4_copy: Connect4, limit: int) -> list[Node]:

        if limit <= 0:
            return [Node(state=connect4_copy, value=self.calculate_state(connect4_copy))]

        states: list[Node] = []
        for possible_drop in connect4_copy.possible_drops():

            new_game_state, result = self.do_move(connect4_copy, possible_drop)
            if result is None:
                node = Node(state=new_game_state, value=self.generate_states(new_game_state, limit=limit - 1))
                states.append(node)
                continue
            else:
                node = Node(state=new_game_state, value=result)
                states.append(node)

        return states

    def get_move(self, connect4_copy: Connect4) -> int:
        """Return number of on among the possible drops based on minmax algorithm."""

        best_moves: list[tuple[int, int]] = []
        for possible_drop in connect4_copy.possible_drops():
            new_game_state, result = self.do_move(connect4_copy, possible_drop)
            if result:
                best_moves.append((possible_drop, result))
            else:
                states = self.generate_states(new_game_state, limit=3)
                best_moves.append((possible_drop, MinMaxAgent.eval_states(states)))

        best_moves.sort(key=lambda move: move[1])
        best_moves.reverse()
        return best_moves[0][0]

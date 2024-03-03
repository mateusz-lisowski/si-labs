import copy
from dataclasses import dataclass
from typing import Optional, TypedDict

from exceptions import AgentException
from connect4 import Connect4


class Node(TypedDict):
    state: Connect4
    value: list['Node'] | int


@dataclass
class Limit:
    limit: int


class MinMaxAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4: Connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.get_move(copy.deepcopy(connect4))

    @staticmethod
    def calculate_state() -> int:
        return 0

    @staticmethod
    def eval_states(states: list[Node]) -> int:
        values: list[int] = []
        for state in states:
            if type(state['value']) is not int:
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

    def generate_states(self, connect4_copy: Connect4, limit: Limit) -> list[Node]:

        if limit.limit <= 0:
            return [Node(state=connect4_copy, value=self.calculate_state())]
        else:
            limit.limit -= 1

        states: list[Node] = []
        for possible_drop in connect4_copy.possible_drops():

            new_game_state, result = self.do_move(connect4_copy, possible_drop)
            if result is None:
                node = Node(state=new_game_state, value=self.generate_states(new_game_state, limit=limit))
                states.append(node)
                continue
            else:
                node = Node(state=new_game_state, value=result)
                states.append(node)

        return states

    def get_move(self, connect4_copy: Connect4) -> int:
        """Return number of on among the possible drops based on minmax algorithm."""

        limit = Limit(10_000)

        best_moves: list[tuple[int, int]] = []
        for possible_drop in connect4_copy.possible_drops():
            new_game_state, result = self.do_move(connect4_copy, possible_drop)
            if result:
                best_moves.append((possible_drop, result))
            else:
                states = self.generate_states(new_game_state, limit)
                best_moves.append((possible_drop, MinMaxAgent.eval_states(states)))

        best_moves.sort(key=lambda move: move[1])
        best_moves.reverse()
        return best_moves[0][0]

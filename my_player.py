from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from game_state_divercite import GameStateDivercite

from seahorse.game.action import Action
from seahorse.game.game_layout.board import Piece  # Ajoute cette ligne pour importer la classe Piece

import time

class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameStateDivercite, remaining_time: int = 1e9, **kwargs) -> Action:
        start_time = time.time()
        best_action = None
        best_value = float('-inf')
        depth = 4  # Fixed shallow depth

        best_action, best_value = self.minimax(current_state, depth, start_time, remaining_time)
        return best_action

    def minimax(self, state, depth, start_time, time_limit, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        if depth == 0 or state.is_done() or time.time() - start_time >= time_limit:
            return None, self.evaluate_state(state)
        
        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            for action in state.generate_possible_heavy_actions():
                next_state = action.get_next_game_state()
                _, eval = self.minimax(next_state, depth - 1, start_time, time_limit, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_action, max_eval
        else:
            min_eval = float('inf')
            best_action = None
            for action in state.generate_possible_heavy_actions():
                next_state = action.get_next_game_state()
                _, eval = self.minimax(next_state, depth - 1, start_time, time_limit, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_action, min_eval

    def evaluate_state(self, state: GameStateDivercite) -> float:
        score = 0
        for player_id, player_score in state.scores.items():
            score += player_score if player_id == self.get_id() else -player_score
            for pos, piece in state.get_rep().get_env().items():
                if piece.get_owner_id() == self.get_id():
                    if piece.get_type()[1] == 'C':
                        score += self.evaluate_city_position(state, pos)
                    else:
                        score += self.evaluate_resource_position(state, pos)
        return score

    def evaluate_city_position(self, state: GameStateDivercite, pos: tuple) -> float:
        neighbors = state.get_neighbours(pos[0], pos[1])
        score = 0
        for neighbor in neighbors.values():
            if isinstance(neighbor[0], Piece):
                if neighbor[0].get_type()[0] == 'R':
                    score += 1
                elif neighbor[0].get_type()[0] == 'C':
                    score -= 1
        return score

    def evaluate_resource_position(self, state: GameStateDivercite, pos: tuple) -> float:
        neighbors = state.get_neighbours(pos[0], pos[1])
        score = 0
        for neighbor in neighbors.values():
            if isinstance(neighbor[0], Piece):
                if neighbor[0].get_type()[0] == 'C':
                    score += 1
        return score


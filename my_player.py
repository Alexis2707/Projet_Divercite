from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from game_state_divercite import GameStateDivercite

from seahorse.game.action import Action
from seahorse.game.game_layout.board import Piece  # Ajoute cette ligne pour importer la classe Piece

class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameStateDivercite, remaining_time: int = 1e9, **kwargs) -> Action:
        def minimax(state, depth, alpha, beta, maximizing_player):
            if depth == 0 or state.is_done():
                return self.evaluate_state(state)
            
            if maximizing_player:
                max_eval = float('-inf')
                for action in state.generate_possible_heavy_actions():
                    next_state = action.get_next_game_state()
                    eval = minimax(next_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float('inf')
                for action in state.generate_possible_heavy_actions():
                    next_state = action.get_next_game_state()
                    eval = minimax(next_state, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval

        best_action = None
        best_value = float('-inf')
        for action in current_state.generate_possible_heavy_actions():
            next_state = action.get_next_game_state()
            value = minimax(next_state, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=False)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def evaluate_state(self, state: GameStateDivercite) -> float:
        score = 0
        for player_id, player_score in state.scores.items():
            # Base score from the game state
            score += player_score if player_id == self.get_id() else -player_score
            
            # Positional advantage
            for pos, piece in state.get_rep().get_env().items():
                if piece.get_owner_id() == self.get_id():
                    if piece.get_type()[1] == 'C':  # City piece
                        score += self.evaluate_city_position(state, pos)
                    else:  # Resource piece
                        score += self.evaluate_resource_position(state, pos)
        return score

    def evaluate_city_position(self, state: GameStateDivercite, pos: tuple) -> float:
        # Example heuristic for city position
        neighbors = state.get_neighbours(pos[0], pos[1])
        score = 0
        for neighbor in neighbors.values():
            if isinstance(neighbor[0], Piece):
                if neighbor[0].get_type()[0] == 'R':  # Resource
                    score += 1
                elif neighbor[0].get_type()[0] == 'C':  # City
                    score -= 1
        return score

    def evaluate_resource_position(self, state: GameStateDivercite, pos: tuple) -> float:
        # Example heuristic for resource position
        neighbors = state.get_neighbours(pos[0], pos[1])
        score = 0
        for neighbor in neighbors.values():
            if isinstance(neighbor[0], Piece):
                if neighbor[0].get_type()[0] == 'C':  # City
                    score += 1
        return score


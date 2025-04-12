from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

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
        # Implémente une fonction d'évaluation pour le plateau
        score = 0
        for player_id, player_score in state.scores.items():
            score += player_score
        return score


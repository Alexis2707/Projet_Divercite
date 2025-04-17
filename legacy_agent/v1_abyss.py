from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from game_state_divercite import GameStateDivercite

from seahorse.game.action import Action
# Ajoute cette ligne pour importer la classe Piece
from seahorse.game.game_layout.board import Piece

import time


class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameStateDivercite, remaining_time: int = 1e9) -> Action:
        start_time = time.time()

        total_moves = current_state.max_step
        moves_left = total_moves - current_state.get_step()

        # On evite la division par 0 de la fin
        if moves_left <= 0:
            moves_left = 1

        if moves_left >= 30:
            time_for_this_move = remaining_time / moves_left
            time_for_this_move = max(0.5, min(time_for_this_move, 60))
        elif moves_left >= 15:
            time_for_this_move = remaining_time * 0.20
        else:
            time_for_this_move = remaining_time / moves_left
            time_for_this_move = max(0.5, min(time_for_this_move, 120))
        # time_for_this_move = remaining_time / moves_left

        best_action = None
        current_depth = 1

        possible_actions = list(
            current_state.generate_possible_heavy_actions())

        # Si peu de temps, retourne une action rapide (ex: la première dans ce cas)
        if time_for_this_move < 0.5:
            return possible_actions[0]

        try:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_for_this_move:
                    break

                best_action_for_depth, _ = self.minimax(
                    current_state,
                    current_depth,
                    start_time,
                    time_for_this_move,
                )
                if best_action_for_depth is not None:
                    best_action = best_action_for_depth

                current_depth += 1
                # Limite de profondeur pour éviter les boucles infinies
                if current_depth > total_moves:
                    break

        # TODO: Definir une exception personalisé TimeLimitExceeded
        except Exception as e:
            print(f"Search interrupted by exception: {e}")

        print("Fin de la recherche à la profondeur", current_depth)
        # Si aucune action n'a été trouvée on prend la première action possible
        if best_action is None:
            best_action = possible_actions[0]

        return best_action

    def minimax(self, state, depth, start_time, time_limit, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            raise TimeoutError("Move time limit exceeded")

        if depth == 0 or state.is_done():
            return None, self.evaluate_state(state)

        best_action = None

        # Amelioration : Trier les actions ici (move ordering) peut améliorer l'élagage alpha-beta
        possible_actions = list(state.generate_possible_heavy_actions())

        if maximizing_player:
            max_eval = float('-inf')
            for action in possible_actions:
                next_state = action.get_next_game_state()
                try:
                    _, eval = self.minimax(
                        next_state, depth - 1, start_time, time_limit, alpha, beta, False)
                except TimeoutError:
                    raise

                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_action, max_eval

        else:
            min_eval = float('inf')
            for action in possible_actions:
                next_state = action.get_next_game_state()
                try:
                    _, eval = self.minimax(
                        next_state, depth - 1, start_time, time_limit, alpha, beta, True)
                except TimeoutError:
                    raise

                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return best_action, min_eval

    def evaluate_state(self, state: GameStateDivercite) -> float:
        opponent_id = [p.get_id()
                       for p in state.players if p.get_id() != self.get_id()][0]

        # Si le jeu est terminé, le score final est la seule chose qui compte
        if state.is_done():
            return state.scores[self.get_id()] - state.scores[opponent_id]

        score_diff = state.scores[self.get_id()] - state.scores[opponent_id]

        state_heuristic = self.calculate_state_heuristic(state, opponent_id)
        W1 = 1.0
        W2 = 0.1
        final_heuristic = W1 * score_diff + W2 * state_heuristic
        return final_heuristic

    def calculate_state_heuristic(self, state: GameStateDivercite, opponent_id: int) -> float:
        value = 0
        for pos, piece in state.get_rep().get_env().items():
            if piece.get_type()[1] == 'C':
                is_mine = piece.get_owner_id() == self.get_id()
                city_value = self.evaluate_city_position(
                    state, pos, piece.get_type()[0])
                value += city_value if is_mine else -city_value
            elif piece.get_type()[1] == 'R':
                resource_value = self.evaluate_resource_position(
                    state, pos, piece.get_type()[0])
                value += resource_value
        return value

    def evaluate_city_position(self, state: GameStateDivercite, pos: tuple, city_color: str) -> float:
        neighbors = state.get_neighbours(pos[0], pos[1])
        neighbor_resources = [n[0] for n in neighbors.values() if isinstance(
            n[0], Piece) and n[0].get_type()[1] == 'R']

        if not neighbor_resources:
            return 0

        resource_colors = set(res.get_type()[0] for res in neighbor_resources)

        if len(resource_colors) == 4 and len(neighbor_resources) == 4:
            return 5  # Divercité
        else:
            same_color_count = sum(
                1 for res in neighbor_resources if res.get_type()[0] == city_color)
            # Amelioration potentielle pour le calcul d'une diversité en cours
            return same_color_count

    """
    TODO:
        - Calcul trop simpliste: prendre en compte quel cite est adjacente à la ressource
        - Ajouter un malus si c'est une ville adverse
        Autres pistes: 
            Calcule le score de cette cité SANS la ressource en 'pos' (difficile à faire parfaitement sans simulation)
            Calcule le score de cette cité AVEC la ressource en 'pos'
            Approche simplifiée : quel est l'effet marginal de CETTE ressource ?
            Est-ce qu'elle complète une Divercité ? Est-ce qu'elle ajoute un point de couleur ?
            Note: La fonction check_divercite existe dans game_state_divercite, mais elle évalue l'état actuel.
            Vous pourriez avoir besoin de simuler l'ajout/retrait pour une évaluation plus précise.
    """

    def evaluate_resource_position(self, state: GameStateDivercite, pos: tuple, ressource_color: str) -> float:
        neighbors = state.get_neighbours(pos[0], pos[1])
        impact = 0
        my_id = self.get_id()

        for neighbor in neighbors.values():
            neighbor_piece = neighbor[0]
            neighbor_pos = neighbor[1]

            if isinstance(neighbor_piece, Piece) and neighbor_piece.get_type()[1] == 'C':
                is_mine = neighbor_piece.get_owner_id() == my_id
                city_color = neighbor_piece.get_type()[0]
                gain = 0
                if ressource_color == city_color:
                    gain += 1
                impact += gain if is_mine else -gain
        return impact

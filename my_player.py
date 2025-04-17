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
from collections import Counter


class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self.W_score = 0.5
        self.W_heuristic = 0.5
        self.W_opening = 0.5

    def compute_action(self, current_state: GameStateDivercite, remaining_time: int = 1e9) -> Action:
        start_time = time.time()

        total_moves = current_state.max_step
        game_progress = total_moves - current_state.get_step()
        moves_left = (total_moves - current_state.get_step()) // 2

        # On evite la division par 0 de la fin
        if moves_left <= 0:
            moves_left = 1

        if game_progress >= 30:
            time_for_this_move = remaining_time / moves_left
            time_for_this_move = max(0.5, min(time_for_this_move, 60))
        else:
            time_for_this_move = remaining_time / moves_left
            time_for_this_move = max(0.5, min(time_for_this_move, 120))

        best_action = None
        # On peut peut-être taffer par la-dessus
        current_depth = 1

        possible_actions = list(
            current_state.generate_possible_heavy_actions())

        # Securite: si peu de temps, retourne une action rapide
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
                if current_depth > game_progress:
                    print("Limite de profondeur atteinte")
                    break

        # On arrete la recherche lorsque l'on remontre l'exception de timeout
        except TimeoutError:
            # logger.info("Fin de la recherche")
            print("Fin de la recherche à la profondeur", current_depth)
            pass

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

        possible_actions = list(state.generate_possible_heavy_actions())
        # Éviter de trier aux feuilles si l'heuristique est déjà calculée
        if depth > 1:
            ordered_actions = self.order_moves(possible_actions, state, maximizing_player)
        else:
            ordered_actions = possible_actions

        if maximizing_player:
            max_eval = float('-inf')
            for action in ordered_actions:
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
            for action in ordered_actions:
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

    def order_moves(self, actions, state, maximizing_player):
        move_scores = []
        my_id = self.get_id()
        opponent_id = next(p.get_id() for p in state.players if p.get_id() != my_id)

        # 1. Calculer le score pour chaque action
        for action in actions:
            # Attention : get_next_game_state() peut être coûteux.
            # Si c'est trop lent, utiliser une heuristique plus rapide ici.
            try:
                next_state = action.get_next_game_state()
                score = next_state.scores[my_id] - next_state.scores[opponent_id]
                move_scores.append((action, score))
            except Exception as e:
                print(f"Erreur lors de l'évaluation de l'action pour le tri: {e}")
                move_scores.append((action, 0 if maximizing_player else 0))

        move_scores.sort(key=lambda item: item[1], reverse=maximizing_player)
        ordered_actions = [item[0] for item in move_scores]
        return ordered_actions

    def evaluate_state(self, state: GameStateDivercite) -> float:
        opponent_id = [p.get_id()
                       for p in state.players if p.get_id() != self.get_id()][0]

        # Si le jeu est terminé, le score final est la seule chose qui compte
        if state.is_done():
            return state.scores[self.get_id()] - state.scores[opponent_id]

        score_diff = state.scores[self.get_id()] - state.scores[opponent_id]

        state_heuristic = self.calculate_state_heuristic(state)

        opening_heuristic_bias = 0
        step = state.get_step()
        if step < 4:
            opening_heuristic_bias = self.calculate_opening_bias(state)

        # W_score = 0.5
        # W_heuristic = 0.5
        # W_opening = 0.5
        # if step >= 34:
        #     W_score = 1.0
        #     W_heuristic = 0.0

        final_heuristic = self.W_score * score_diff
        final_heuristic += self.W_heuristic * state_heuristic
        final_heuristic += self.W_opening * opening_heuristic_bias
        return final_heuristic

    def calculate_opening_bias(self, state: GameStateDivercite) -> float:
        """Calcule un bonus basé sur la proximité des pièces du joueur au centre."""
        bias = 0
        center_pos = (4, 4)
        for pos, piece in state.get_rep().get_env().items():
            if piece.get_owner_id() == self.get_id():
                distance = abs(pos[0] - center_pos[0]) + abs(pos[1] - center_pos[1])
                bias += max(0, 4 - distance)
        return bias

    def calculate_state_heuristic(self, state: GameStateDivercite) -> float:
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

        if state.check_divercite(pos):
            return 5
        else:
            resource_color_counts = Counter(res.get_type()[0] for res in neighbor_resources)
            final_city_value = 0.0
            same_color_count = resource_color_counts[city_color]

            # On cherche les ressources de couleur différente de la cité
            # présent au moins 2 fois et on applique une pénalité sur le
            # positionnement de la cité
            double_wrong_color = False
            for res_color, count in resource_color_counts.items():
                if res_color != city_color and count >= 2:
                    double_wrong_color = True
                    break
            if double_wrong_color:
                final_city_value -= 0.5
            elif resource_color_counts[city_color] < 2:
                unique_colors_count = len(resource_color_counts)
                diversity_bonus = 0.1 * unique_colors_count
                final_city_value += diversity_bonus
            final_city_value += same_color_count
            return final_city_value

    """
    TODO:
    Autres pistes:
        Calcule le score de cette cité SANS la ressource en 'pos' (difficile à
        faire parfaitement sans simulation)
        Calcule le score de cette cité AVEC la ressource en 'pos'
        Approche simplifiée : quel est l'effet marginal de CETTE ressource ?
        Est-ce qu'elle complète une Divercité ? Est-ce qu'elle ajoute un point
        de couleur ?
        Note: La fonction check_divercite existe dans game_state_divercite,
        mais elle évalue l'état actuel.
        Vous pourriez avoir besoin de simuler l'ajout/retrait pour une
        évaluation plus précise.
    """

    def evaluate_resource_position(self, state: GameStateDivercite, pos: tuple, ressource_color: str) -> float:
        neighbors = state.get_neighbours(pos[0], pos[1])
        impact = 0
        my_id = self.get_id()

        # Pour chaque voisin de la ressource, on regarde si il y a une cite
        # Si il y a une cite, on regarde a qui elle appartient
        # On evalue l'impact de la ressource sur la cite:
        # - Si la ressource est de la même couleur que la cite, on ajoute 1
        # - Si la ressource est de couleur différente, on regarde si il y a au
        # moins 2 ressources de cette couleur, si oui on applique un malus sur
        # le gain de la ressource
        for neighbor in neighbors.values():
            neighbor_piece = neighbor[0]
            neighbor_pos = neighbor[1]

            if isinstance(neighbor_piece, Piece) and neighbor_piece.get_type()[1] == 'C':
                is_mine = neighbor_piece.get_owner_id() == my_id
                city_color = neighbor_piece.get_type()[0]
                gain = 0

                # On recupère les ressources autour de la cité voisine
                city_neighbors = state.get_neighbours(
                        neighbor_pos[0], neighbor_pos[1])
                city_neighbor_resources = [
                        n[0] for n in city_neighbors.values()
                        if isinstance(n[0], Piece)
                        and n[0].get_type()[1] == 'R']

                # On verifie les voisins autour de la cité
                resource_color_counts = Counter(
                        res.get_type()[0] for res in city_neighbor_resources)

                double_wrong_color = False
                for res_color, count in resource_color_counts.items():
                    if res_color != city_color and count >= 2 and res_color == ressource_color:
                        double_wrong_color = True
                        break
                # Si on a 2 voisins de la même couleur et de couleur différente
                # de la cité on applique un malus
                if double_wrong_color:
                    gain -= 0.5
                # Si pas de malus, alors on peut bonifier les resources de
                # couleur uniques autour d'une ville, on applique un bonus de
                # diversite pour la ressource que l'on va ajouter
                elif resource_color_counts[city_color] < 2:
                    diversity_bonus = 0.1
                    gain += diversity_bonus

                # Sinon juste on compte un point en plus de la couleur qui
                # match
                if ressource_color == city_color:
                    gain += 1
                impact += gain if is_mine else -gain
        return impact

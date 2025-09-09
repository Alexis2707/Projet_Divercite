# Projet Divercite

## Overview

Projet Divercite is a Python-based repository primarily focused on the development of `my_player.py`, an automated player for the board game "Divercite." This file contains advanced strategies and algorithms to compete effectively in the game. 

In addition to the automated player, this repository also includes a complete implementation of the Divercite board game, allowing users to play and experiment with various gameplay configurations.

The `rapport.pdf` file provides a detailed explanation of the strategies and algorithms used in `my_player.py`, along with insights into the game's mechanics and design choices.
The `Projet_Divercite_H25.pdf` file contains the project description and requirements.

## Features

- Implementation of the game's board and state mechanics.
- Support for various types of players:
  - Random player.
  - Greedy player.
  - Custom player with advanced strategies (`my_player.py`).
- Configurable gameplay modes:
  - Local play.
  - Human vs Computer.
  - Human vs Human.
  - Remote multiplayer.
- GUI support for interactive gameplay.
- Logging and state recording functionality.

## Requirements

The project requires Python and the dependencies listed in the `requirements.txt` file. Install them using pip:

```bash
pip install -r requirements.txt
```

## Dependencies

- `aiohttp`
- `colorama`
- `loguru`
- `seahorse`
- ... and more (refer to `requirements.txt`).

## How to Run

### 1. Local Play

Run the game locally by specifying two player implementations:

```bash
python main_divercite.py -t local players_list [path_to_player1.py] [path_to_player2.py]
```

### 2. Human vs Computer

Launch the GUI to challenge your player:

```bash
python main_divercite.py -t human_vs_computer players_list [path_to_player.py]
```

### 3. Remote Multiplayer

Host a game or connect to a remote game using the following options:

- Host a game:

```bash
python main_divercite.py -t host_game -a [external_ip] players_list [path_to_player.py]
```

- Connect to a game:

```bash
python main_divercite.py -t connect -a [host_ip] players_list [path_to_player.py]
```

### 4. Human vs Human

Launch a local GUI for two players to experiment with the game's mechanics:

```bash
python main_divercite.py -t human_vs_human
```

## Code Structure

- `board_divercite.py`: Contains the implementation of the game board and its mechanics.
- `game_state_divercite.py`: Represents the state of the game and includes logic for validating moves and scoring.
- `player_divercite.py`: Provides the base class for players.
- `my_player.py`: The main focus of this repository, which implements a custom automated player with advanced strategies.
- `random_player_divercite.py`, `greedy_player_divercite.py`: Implement different player strategies.
- `master_divercite.py`: Manages the overall game and determines the winner based on scores.
- `main_divercite.py`: Entry point for running the game.

---
Enjoy playing Divercite!

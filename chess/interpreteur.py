import DQN
import torch
import torch.nn as nn
import numpy as np
import random


dqn = DQN.reseau_neurones

input_size = 64
hidden_size = 128
output_size = 64
# Création du réseau de neurones
net = dqn(input_size, hidden_size, output_size)

# Charger les poids du modèle entraîné
net.load_state_dict(torch.load('q_learning_chess_model.pth'))
net.eval()  # Mettre le modèle en mode évaluation


def BestMove(game_state, valid_moves, return_queue):
    board = game_state.board
    # À l'intérieur de la boucle de jeu
    state = np.array(DQN.board_to_array(board)).reshape(1, -1)
    state_tensor = torch.FloatTensor(state)

    with torch.no_grad():
        q_values = net(state_tensor)

    action = q_values.argmax().item()  # Choisir l'action avec la plus grande valeur Q
    q_values[0][action] = -10000

    if action < len(valid_moves):
        move = valid_moves[action]
        return_queue.put(move)
    else:
        while action >= len(valid_moves):
            action = q_values.argmax().item()  # Choisir l'action avec la plus grande valeur Q
            q_values[0][action] = -10000
            # print("Action invalide:", action)
        move = valid_moves[action]
        return_queue.put(move)

def findRandomMove(valid_moves):
    """
    Picks and returns a random valid move.
    """
    return random.choice(valid_moves)

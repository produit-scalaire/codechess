import reseaux_neurones
import torch
import numpy as np
import chessEngine

game_state = chessEngine.GameState()

pieces = ['', 'R', 'B', 'N', 'K', 'Q']
colonne = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
all_possible_move = []
for piece in pieces:
    for row in range (8):
        for col in colonne:
            all_possible_move.append(piece+col +str(row+1))

#transformer l'echiquier en un tableua numpy de nombre
piece_score = {"K": 20, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1, "-": 0}
def board_to_array(board):
    array = []
    for row in board:
        for piece in row:
            if piece[0] == "--":
                color_factor = 0
            if piece[0] == "w":
                color_factor = 1
            else:
                color_factor = -1
            array.append(color_factor * piece_score[piece[1]])
    return array


def filter_legal_moves(q_values, valid_moves):       #list_of_moves est la liste des valeur en sortie du reseau de neurones
    moves = []
    valid_moves_str = []
    for move in valid_moves:
        valid_moves_str.append(str(move))

    for index_move in range (len(valid_moves_str)):
        move = valid_moves_str[index_move]
        if move in all_possible_move:
            moves.append([valid_moves[index_move], q_values[0][all_possible_move.index(move)]])
        elif len(move) == 3 and move[0] + move[-2] + move[-1] in all_possible_move:
            moves.append([valid_moves[index_move], q_values[0][all_possible_move.index(move[0] + move[-2] + move[-1])]])
    return moves

def argmax(action_space):
    i = 0
    max = -10000
    index = 0
    for action in action_space:
        if action[1] > max:
            max = action[1]
            index = i
        i +=  1
    return index
def select_action(q_values, valid_moves):

    action_space = filter_legal_moves(q_values, valid_moves)
    action = argmax(action_space)
    move = valid_moves[action]
    return move

# taille des couches
input_size = 64
hidden_size1 = 64 * 4
hidden_size2 = 300

#le reseau choisit la pièce à bouger avec la case soit 6 * 64 + 2 choix
output_size = 384+2

cnn = reseaux_neurones.chess_neural_network(input_size, hidden_size1, hidden_size2, output_size)
cnn.eval()

# Charger les poids du modèle entraîné
cnn.load_state_dict(torch.load('models/CNN_chess.pth'))
cnn.eval()  # Mettre le modèle en mode évaluation

def BestMove(game_state, valid_moves, return_queue):
    board = game_state.board
    # À l'intérieur de la boucle de jeu
    state = np.array(board_to_array(board)).reshape(1, -1)
    state_tensor = torch.FloatTensor(state)

    with torch.no_grad():
        q_values = cnn(state_tensor)

    move = select_action(q_values, valid_moves)
    return_queue.put(move)


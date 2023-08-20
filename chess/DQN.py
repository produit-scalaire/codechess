import numpy as np
import chessEngine
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand

game_state = chessEngine.GameState()

piece_score = {"K": 20, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1, "-": 0}

CHECKMATE = 1000
STALEMATE = 0

knight_scores = [[0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0],
                 [0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1],
                 [0.2, 0.5, 0.6, 0.65, 0.65, 0.6, 0.5, 0.2],
                 [0.2, 0.55, 0.65, 0.7, 0.7, 0.65, 0.55, 0.2],
                 [0.2, 0.5, 0.65, 0.7, 0.7, 0.65, 0.5, 0.2],
                 [0.2, 0.55, 0.6, 0.65, 0.65, 0.6, 0.55, 0.2],
                 [0.1, 0.3, 0.5, 0.55, 0.55, 0.5, 0.3, 0.1],
                 [0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0]]

bishop_scores = [[0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
                 [0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2],
                 [0.2, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.2],
                 [0.2, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.2],
                 [0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.4, 0.2],
                 [0.2, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2],
                 [0.2, 0.5, 0.4, 0.4, 0.4, 0.4, 0.5, 0.2],
                 [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0]]

rook_scores = [[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
               [0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.25]]

queen_scores = [[0.0, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.0],
                [0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2],
                [0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.2],
                [0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3],
                [0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3],
                [0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.2],
                [0.2, 0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.2],
                [0.0, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.0]]

pawn_scores = [[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
               [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
               [0.3, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.3],
               [0.25, 0.25, 0.3, 0.45, 0.45, 0.3, 0.25, 0.25],
               [0.2, 0.2, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2],
               [0.25, 0.15, 0.1, 0.2, 0.2, 0.1, 0.15, 0.25],
               [0.25, 0.3, 0.3, 0.0, 0.0, 0.3, 0.3, 0.25],
               [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]

piece_position_scores = {"wN": knight_scores,
                         "bN": knight_scores[::-1],
                         "wB": bishop_scores,
                         "bB": bishop_scores[::-1],
                         "wQ": queen_scores,
                         "bQ": queen_scores[::-1],
                         "wR": rook_scores,
                         "bR": rook_scores[::-1],
                         "wp": pawn_scores,
                         "bp": pawn_scores[::-1]}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scoreBoard(game_state):
    """
    Score the board. A positive score is good for white, a negative score is good for black.
    """
    if game_state.checkmate:
        if game_state.white_to_move:
            return -CHECKMATE  # black wins
        else:
            return CHECKMATE  # white wins
    elif game_state.stalemate:
        return STALEMATE
    score = 0
    for row in range(len(game_state.board)):
        for col in range(len(game_state.board[row])):
            piece = game_state.board[row][col]
            if piece != "--":
                piece_position_score = 0
                if piece[1] != "K":
                    piece_position_score = piece_position_scores[piece][row][col]
                if piece[0] == "w":
                    score += piece_score[piece[1]] + piece_position_score
                if piece[0] == "b":
                    score -= piece_score[piece[1]] + piece_position_score

    return score


def game_is_over(board):
    L = []
    for row in board:
        for piece in row:
            if piece != "--":
                L.append(piece)
    if set(L) == set(["wK", "bK"]):
        draw = True
    elif set(L) == set(["wK", "bK", "wB", "bB"]):
        draw = True
    elif set(L) == set(["wK", "bK", "wN", "bN"]):
        draw = True
    elif set(L) == set(["wK", "bK", "wN"]):
        draw = True
    elif set(L) == set(["wK", "bK", "bN"]):
        draw = True
    elif set(L) == set(["wK", "bK", "wB"]):
        draw = True
    elif set(L) == set(["wK", "bK", "bB"]):
        draw = True
    else:
        draw = False
    if game_state.stalemate or game_state.checkmate or draw:
        return True

    else:
        return False

# Fonction pour convertir le plateau d'échecs en tableau numpy
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

# Fonction pour sélectionner une action avec epsilon-greedy
def select_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        index = np.random.randint(q_values.size()[1])
        q_values[0][index] = -10000
        return index, q_values
    else:
        index= q_values.max(1)[1]
        q_values[0][index.item()] = -10000
        return index.item(), q_values


# Fonction pour obtenir la récompense en fonction de l'état du plateau
def get_reward(board):
    # Utilisez une logique plus avancée pour calculer la récompense en fonction de l'état du jeu
    if game_state.checkmate:
        return -1000
    elif game_state.stalemate:
        return 0
    else:
        return 1000
#fonction qui calcul le epsilon en fonction de l'iteration
def eps(num_episode, epsilon_start, epsilon_end):
    return epsilon_start * np.exp(np.log(epsilon_end/epsilon_start)/num_episode)

# Création du réseau de neurones
class reseau_neurones(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(reseau_neurones, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Configuration de l'environnement
board = game_state.board

# Configuration du réseau de neurones
input_size = 64  # 64 cases d'échecs
hidden_size = 128
output_size = 64  # Nombre de coups possibles
net = reseau_neurones(input_size, hidden_size, output_size).to(device)

# Définition de l'algorithme d'optimisation
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Paramètres d'apprentissage par renforcement
gamma = 0.9  # Facteur d'actualisation
epsilon_start = 0.90  # Exploration vs exploitation au debut
epsilon_end = 0.01

# Entraînement par renforcement
def Train():
    num_episodes = 100
    i = 0
    for episode in range(num_episodes):
        v = 0
        epsilon = eps(num_episodes, epsilon_start, epsilon_end)
        while not game_is_over(board) or v < 100:
            state = np.array(board_to_array(board)).reshape(1, -1)
            state_tensor = torch.FloatTensor(state)

            with torch.no_grad():
                q_values = net(state_tensor)
            action = select_action(q_values, epsilon)[0]
            q_values = select_action(q_values, epsilon)[1]

            # Jouer le coup sur le plateau
            legal_moves = list(game_state.getValidMoves())
            if action < len(legal_moves):
                move = legal_moves[action]
                game_state.makeMove(move)
                reward = 10
            elif len(legal_moves) != 0:
                while action >= len(legal_moves):
                    print("here")
                    action = select_action(q_values, epsilon)[0]
                    q_values = select_action(q_values, epsilon)[1]
                #print("Action invalide:", action)
                move = legal_moves[action]
                game_state.makeMove(move)
                reward = -10
            else:
                break
            # Calculer la récompense
            reward += scoreBoard(game_state)

            new_state = np.array(board_to_array(board)).reshape(1, -1)
            new_state_tensor = torch.FloatTensor(new_state)

            with torch.no_grad():
                new_q_values = net(new_state_tensor)
            max_new_q, _ = torch.max(new_q_values, dim=1)
            target_q = reward + gamma * max_new_q

            q_values[0, action] = target_q

            # Mise à jour du réseau de neurones
            optimizer.zero_grad()
            loss = nn.MSELoss()(net(state_tensor), q_values)
            loss.backward()
            optimizer.step()
            v +=1
        i += 1
        if i % 1 == 0:
            print(i)

        game_state.reset_game()

    print("Entraînement terminé.")
# Enregistrement du modèle entraîné
#Train()
torch.save(net.state_dict(), 'q_learning_chess_model.pth')

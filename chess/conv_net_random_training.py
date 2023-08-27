import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import chessEngine
import reseaux_neurones

device = "cuda" if torch.cuda.is_available() else "cpu"

#data
f = open('games.csv', 'r')
les_lignes = f.readlines()
f.close()
chess_data = []
for ligne in les_lignes:
    ligne = ligne.strip()
    chess_data.append(ligne.split('\n'))
chess_data.pop(0)
chess_games = []
for game in chess_data:
    game = game[0].strip()
    chess_games.append(game.split(','))
games = []
for game in chess_games:
    if int(game[11]) >= 2000:
        games.append(game[12])
moves = []
for game in games:
    game = game.strip()
    moves.append(game.split(' '))

#print(moves[0])
#print(len(moves))
# taille des couche
input_size = 64
hidden_size1 = 64 * 4
hidden_size2 = 300

#le reseau choisit la pièce à bouger avec la case soit 6 * 64 + 2 choix
output_size = 384+2

pieces = ['', 'R', 'B', 'N', 'K', 'Q']
colonne = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
all_possible_move = []
for piece in pieces:
    for row in range (8):
        for col in colonne:
            all_possible_move.append(piece+col +str(row+1))

all_possible_move.append('O-O')
all_possible_move.append('O-O-O')
#print(all_possible_move)
#print(len(all_possible_move))

#model sur device
net = reseaux_neurones.chess_neural_network(input_size, hidden_size1, hidden_size2, output_size).to(device)

#loss fct
loss_MSE = nn.MSELoss()

#optimizer, lr = learning rate
optimizer = torch.optim.SGD(params = net.parameters(), lr = 0.01)

#état du jeu
game_state = chessEngine.GameState()

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

print(board_to_array(game_state.board))
print(game_state.getValidMoves()[0].moveID)
#Training loop
def Train():
    num_epoch = 20
    i = 0
    for epoch in range (num_epoch):
        for game in moves:
            v = 0
            while v < 100 or not game_state.checkmate or not game_state.game_is_over or not game_state.draw or not game_state.stalemate:
                print(game_state.checkmate)

                valid_move = game_state.getValidMoves()
                if len(game) <= v or len(valid_move) == 0:
                    break
                if not game_state.white_to_move :   #on entraine le reseau avec les noirs
                    #on met le model en mode training
                    net.train()

                    # transforme l'echiquier en qlq ch de comprehensible pour le reseau
                    board = game_state.board
                    state = np.array(board_to_array(board)).reshape(1, -1)
                    state_tensor = torch.FloatTensor(state)

                    #1 on utilise la "forward methode"
                    y_pred = net(state_tensor)

                    # on trouve l'argument du coup et on crée un vecteur pour comparer avec la prediction
                    #print(game)
                    #print(f"v = {v}")
                    #print(len(game))
                    print(game[v])
                    first_condition = False
                    if game[v][-1] == '#':
                        break
                    elif len(game[v]) > 2 :
                        if game[v][2] == "=":
                            if game[v][1] == "x":
                                if game[v][-1] == "+":
                                    index = all_possible_move.index(game[v][2:-3])
                                    y_train = [0] * 386
                                    y_train[index] = 1
                                else:
                                    index = all_possible_move.index(game[v][2:-2])
                                    y_train = [0] * 386
                                    y_train[index] = 1
                            else:
                                if game[v][-1] == "+":
                                    index = all_possible_move.index(game[v][:-3])
                                    y_train = [0] * 386
                                    y_train[index] = 1
                                else:
                                    index = all_possible_move.index(game[v][:-2])
                                    y_train = [0] * 386
                                    y_train[index] = 1
                            first_condition = True
                        elif len(game[v]) > 4:
                            if game[v][4] == "=":
                                if game[v][1] == "x":
                                    if game[v][-1] == "+":
                                        index = all_possible_move.index(game[v][2:-3])
                                        y_train = [0] * 386
                                        y_train[index] = 1
                                    else:
                                        index = all_possible_move.index(game[v][2:-2])
                                        y_train = [0] * 386
                                        y_train[index] = 1
                                    first_condition = True
                                else:
                                    if game[v][-1] == "+":
                                        index = all_possible_move.index(game[v][:-3])
                                        y_train = [0] * 386
                                        y_train[index] = 1
                                    else:
                                        index = all_possible_move.index(game[v][:-2])
                                        y_train = [0] * 386
                                        y_train[index] = 1
                                    first_condition = True

                    if game[v][0] in pieces and not first_condition:
                        if game[v][-1] == '+':
                            index = all_possible_move.index(game[v][0] + game[v][-3] + game[v][-2])
                            y_train = [0] * 386
                            y_train[index] = 1
                        else:
                            index = all_possible_move.index(game[v][0] + game[v][-2] + game[v][-1])
                            y_train = [0] * 386
                            y_train[index] = 1
                    elif game[v][0] in colonne and game[v][1] == "x" and not first_condition:
                        if game[v][-1] == "+":
                            index = all_possible_move.index(game[v][2:-1])
                            y_train = [0] * 386
                            y_train[index] = 1
                        else:
                            index = all_possible_move.index(game[v][2:])
                            y_train = [0] * 386
                            y_train[index] = 1
                    elif game[v][0] == "O" and not first_condition:
                        if game[v][-1] == "+":
                            index = all_possible_move.index(game[v][:-1])
                            y_train = [0] * 386
                            y_train[index] = 1
                        else:
                            index = all_possible_move.index(game[v])
                            y_train = [0] * 386
                            y_train[index] = 1

                    elif game[v][-1] == '+' and not first_condition:
                        index = all_possible_move.index(game[v][0] + game[v][1])
                        y_train = [0] * 386
                        y_train[index] = 1

                    else:
                        if not first_condition:
                            index = all_possible_move.index(game[v][-2] + game[v][-1])
                            y_train = [0] * 386
                            y_train[index] = 1

                    #on modifie le type de y_train
                    y_train = torch.tensor([y_train], dtype=torch.float32)
                    #on joue sur l'echiquier le coup
                    index = 0
                    #print(valid_move_str)
                    print(game[v])
                    #print(game_state.board)
                    #print(game_state.white_to_move)
                    #print(game)
                    if game[v][-1] == '+':
                        game[v] = game[v][:-1]
                    if game[v][0] == 'N' and game[v][1] in colonne and (len(game[v]) == 4 or len(game[v]) == 5):
                        game[v] = game[v][0] + game[v][2:]
                    #print('here')
                    #print(len(valid_move))
                    while game[v] != str(valid_move[index]):
                        #print(index)
                        if index + 1 == len(valid_move):
                            break
                        index += 1
                    #print(game[v])
                    game_state.makeMove(valid_move[index])  # on joue le coup

                    #2 on calcule le loss
                    loss = loss_MSE(y_pred, y_train)

                    #3 zero grad
                    optimizer.zero_grad()

                    #4 vague de propagation
                    loss.backward()

                    #5 on met à jour le reseau
                    optimizer.step()

                    v += 1
                    i += 1

                    net.eval()

                else:
                    index = 0
                    #print(valid_move_str)
                    #print(game[v])
                    print(f"index game {moves.index(game)}")

                    if game[v][-1] == '+':
                        game[v] = game[v][:-1]
                    if game[v][0] == 'N' and game[v][1] in colonne and (len(game[v]) == 4 or len(game[v]) == 5):
                        game[v] = game[v][0] + game[v][2:]
                        print(game[v])
                    #print(len(valid_move))
                    while game[v] != str(valid_move[index]):

                        #print(index)
                        #print(len(valid_move))
                        if index + 1 == len(valid_move):
                            break
                        index += 1
                    game_state.makeMove(valid_move[index]) #on joue le coup

                    v += 1
                    i += 1

                    net.eval()

                print(f"i = {i}")

            game_state.reset_game()
        print(f"epoch = {epoch}")

    print(f"L'entrainement est terminé: {num_epoch}")

Train()
#sauvegarder le model
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#cree le chemin de la sauvegarde du model
MODEL_NAME = "CNN_chess.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#sauvegarder le model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=net.state_dict(), f=MODEL_SAVE_PATH)


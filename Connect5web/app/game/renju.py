import sqlite3
import torch
from flask import Blueprint, request, jsonify

from app.game.backened.game import Board
from app.game.backened.mcts_alphazero import MCTSPlayer
from app.game.backened.policy_value_net import PolicyValueNet

renju_api = Blueprint('renju_api', __name__)

# import sys
#
# root = Path(__file__).parent.parent.parent
# sys.path.append(str(root))
# print(sys.path)


EMPTY = -1
BLACK = 0
WHITE = 1
board_width = 15
board_height = 15


def transformerTo(chesshistory):
    board = Board(idth=15, height=15, n_in_row=5)
    board.init_board()

    for history in chesshistory:
        x = history['x']
        y = history['y']
        move = board.location_to_move([x, y])
        # print(move)
        board.do_move(move)
    return board


@renju_api.route("/getMove", methods=["POST", "GET"])
def getMove():
    chesshistory = request.json.get("chesshistory")
    board = transformerTo(chesshistory)
    model = torch.jit.load("./game/model/20b128c_renju.pt", map_location=torch.device("cpu")).to(torch.device("cpu"))
    policy_value_net = PolicyValueNet(board_width, board_height, None, use_gpu=True, best_model=model)
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                             c_puct=3.0,
                             n_playout=100,
                             is_selfplay=0)
    move = mcts_player.get_action(board)
    x, y = board.move_to_location(move)

    board.do_move(move)
    has_winner, winner = board.has_a_winner()
    if winner == 1:
        winner = BLACK
    else:
        winner = WHITE
    if has_winner:
        has_winner = 1
    else:
        has_winner = 0
    return jsonify({"x": int(x), "y": int(y), "has_winner": int(has_winner), "winner": int(winner)})


@renju_api.route("/getWinner", methods=["POST", "GET"])
def getWinner():
    chesshistory = request.json.get("chesshistory")
    board = transformerTo(chesshistory)
    x, y = request.json.get('x'),request.json.get('y')
    move = Board().location_to_move((x,y))
    board.do_move(move)
    has_winner, winner = board.has_a_winner()
    if winner == 1:
        winner = BLACK
    else:
        winner = WHITE
    if has_winner:
        has_winner = 1
    else:
        has_winner = 0
    return jsonify({"x": int(x), "y": int(y), "has_winner": int(has_winner), "winner": int(winner)})

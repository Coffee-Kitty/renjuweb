import json
import random
import time
from datetime import datetime

from flask import Blueprint, request, jsonify

from app.database import get_mysql_connection
from app.game.backened.game import Board
from app.game.renju import transformerTo, BLACK, WHITE

pk_api = Blueprint('pk_api', __name__)


@pk_api.route('/matchPlayers', methods=['POST'])
def match_players():
    data = request.get_json()
    current_player_name = data.get('name')
    print('match-------------------------')

    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT state FROM User WHERE username=%s', (current_player_name,))
    state = cursor.fetchone()['state']

    if state == 'isPK':
        cursor.execute(
            'SELECT id, player1_name, player2_name FROM Game WHERE (player1_name=%s OR player2_name=%s) AND winner_name IS NULL AND ended_at IS NULL',
            (current_player_name, current_player_name)
        )
        result = cursor.fetchone()
        if result:
            game_id = result['id']
            player1_name = result['player1_name']
            player2_name = result['player2_name']
            conn.close()
            return jsonify(
                {"message": "match success", 'name': current_player_name, 'game_id': game_id, "player1": player1_name,
                 "player2": player2_name}), 200

    cursor.execute("UPDATE User SET state=%s WHERE username=%s", ('seek', current_player_name))

    matched_players = False
    while not matched_players:
        cursor.execute("SELECT username FROM User WHERE state=%s AND username!=%s", ('seek', current_player_name))
        matched_players = cursor.fetchall()
        conn.close()
        time.sleep(1)
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT state FROM User WHERE username=%s', (current_player_name,))
        state = cursor.fetchone()['state']
        if state == 'isPK':
            cursor.execute(
                'SELECT id, player1_name, player2_name FROM Game WHERE (player1_name=%s OR player2_name=%s) AND winner_name IS NULL AND ended_at IS NULL',
                (current_player_name, current_player_name)
            )
            result = cursor.fetchone()
            if result:
                game_id = result['id']
                player1_name = result['player1_name']
                player2_name = result['player2_name']
                conn.close()
                return jsonify(
                    {"message": "match success", 'name': current_player_name, 'game_id': game_id,
                     "player1": player1_name,
                     "player2": player2_name}), 200

    if matched_players:
        matched_player_name = random.choice(matched_players)['username']
        cursor.execute("UPDATE User SET state=%s WHERE username=%s OR username=%s",
                       ('isPK', current_player_name, matched_player_name))

        cursor.execute('SELECT MAX(id) AS max_id FROM Game')
        ids = cursor.fetchone()['max_id'] + 1
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
        INSERT INTO Game (id, player1_name, player2_name,created_at)
        VALUES (%s, %s, %s,%s)
        ''', (ids, current_player_name, matched_player_name, created_at))

        conn.close()
        return jsonify(
            {"message": "match success", 'name': current_player_name, 'game_id': ids, "player1": current_player_name,
             "player2": matched_player_name}), 200

    else:
        conn.close()
        return jsonify({"message": "no matched players"}), 404


@pk_api.route('/getMove', methods=['POST'])
def getMove():
    data = request.get_json()
    print(data)
    player = data.get('player')
    game_id = data.get('game_id')
    player1_name = data.get('player1_name')
    player2_name = data.get('player2_name')
    name = data.get('name')
    chesshistory = data.get('chesshistory')

    conn = get_mysql_connection()
    cursor = conn.cursor()

    if player == 'player1':
        x = data.get('x')
        y = data.get('y')
        move = Board().location_to_move([x, y])
        move_number = len(chesshistory)

        cursor.execute("SELECT move FROM Move WHERE game_id=%s AND move_number=%s", (game_id, move_number))
        result = cursor.fetchone()
        if not result:
            cursor.execute('''
                   INSERT INTO Move (game_id, player_name, move, move_number)
                   VALUES (%s, %s, %s, %s)
                   ''', (game_id, name, move, move_number))
            cursor.execute("""
                    INSERT INTO Train_data (game_id,board,move,move_number,used,win)
                    VALUES (%s,%s,%s,%s,%s,%s)
            """,(game_id,json.dumps(chesshistory),move,move_number,"false","false"))

        board = transformerTo(chesshistory)
        board.do_move(move)
        has_winner, winner = board.has_a_winner()
    else:
        move_number = len(chesshistory) + 1
        board = transformerTo(chesshistory)
        move = False
        while not move:
            cursor.execute("SELECT move FROM Move WHERE game_id=%s AND move_number=%s", (game_id, move_number))
            result = cursor.fetchone()
            if result:
                move = result['move']
            else:
                time.sleep(0.5)

        x, y = board.move_to_location(move)
        board.do_move(move)
        has_winner, winner = board.has_a_winner()

    if winner == 1:
        winner = BLACK
        winner_name = player1_name
    else:
        winner = WHITE
        winner_name = player2_name

    if has_winner:
        cursor.execute('''
               UPDATE Game SET winner_name=%s, ended_at=NOW() WHERE id=%s
               ''', (winner_name, game_id))
        cursor.execute("UPDATE User SET state=%s WHERE username=%s", ('online', player1_name))
        cursor.execute("UPDATE User SET state=%s WHERE username=%s", ('online', player2_name))
        if winner == BLACK:
            cursor.execute("UPDATE Train_data SET win=%s WHERE game_id=%s and mod(move_number, 2) = 1", ("true", game_id))
        else:
            cursor.execute("UPDATE Train_data SET win=%s WHERE game_id=%s and mod(move_number, 2) = 0",("true", game_id))


    conn.close()

    return jsonify({"x": int(x), "y": int(y), "has_winner": int(has_winner), "winner": int(winner)})



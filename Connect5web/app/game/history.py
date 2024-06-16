from flask import Blueprint, jsonify,request

from app.database import get_mysql_connection
from app.game.backened.game import Board

history_api = Blueprint('history_api',__name__)


# Retrieve a single game history entry by id
@history_api.route('/getHistory', methods=['GET','POST'])
def get_history_by_name():
    data = request.get_json()
    name = data.get('name')
    cur = get_mysql_connection().cursor()
    cur.execute("SELECT * FROM Game WHERE player1_name = %s or player2_name= %s", (name,name))
    history = cur.fetchall()
    cur.close()
    if history:
        return jsonify(history)
    else:
        return jsonify({'message': 'Entry not found'}),205


# Delete a game history entry by id
@history_api.route('/deleteHistory', methods=['GET','POST'])
def delete_history():
    data = request.get_json()
    id = data.get('id')
    cur = get_mysql_connection().cursor()
    cur.execute("DELETE FROM Game WHERE id = %s", (id,))
    cur.close()
    return jsonify({'message': 'Game history entry deleted'})


@history_api.route('/getMemory', methods=['GET','POST'])
def getMemory():
    data = request.get_json()
    id = data.get('id')
    cur = get_mysql_connection().cursor()


    cur.execute("SELECT * FROM Game WHERE id=%s", (id,))
    history = cur.fetchone()
    cur.execute("SELECT move FROM MOVE WHERE game_id=%s ORDER BY move_number ASC",(id,))
    moves = cur.fetchall()

    board = Board(idth=15, height=15, n_in_row=5)
    board.init_board()
    moves = [board.move_to_location(data['move']) for data in moves]
    print(moves)
    if not history or not moves:
        return jsonify({'message': 'Entry not found'}), 404
    cur.close()
    return jsonify({"moves":moves,"history":history})


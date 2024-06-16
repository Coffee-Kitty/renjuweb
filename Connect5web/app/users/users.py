import os

import pymysql

from flask import Blueprint, request, jsonify

from app.database import get_mysql_connection

users_api = Blueprint('users_api', __name__)

@users_api.route("/getByName", methods=["POST", "GET"])
def getByName():
    data = request.get_json()
    username = data.get('name')

    conn = get_mysql_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)  # 设置返回结果为字典形式，方便操作

    cursor.execute("SELECT * FROM User WHERE username=%s", (username,))
    user = cursor.fetchone()

    if user:
        user_info = {
            "id":user['id'],
            "name": user['username'],
            "email": user['email'],
            "registration_time": user['registration_time'],
            "avatar": user['avatar'],
            "state": user['state']
        }
        return jsonify(user_info), 200
    else:
        return jsonify({"message": "user not found"}), 404

@users_api.route("/updateUserById", methods=["POST", "GET"])
def updateUserById():
    data = request.get_json()
    user_id = data.get('id')
    name = data.get('name')
    email = data.get('email')
    state = data.get('state')
    avatar=data.get('avatar')

    conn = get_mysql_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)  # 设置返回结果为字典形式，方便操作

    # Check if the user exists
    cursor.execute("SELECT * FROM User WHERE id = %s", (user_id,))
    user = cursor.fetchone()

    if user:
        # Update the user information
        cursor.execute("""
            UPDATE User 
            SET username = %s, email = %s, avatar = %s,state = %s
            WHERE id = %s
        """, (name, email, avatar,state,user_id))
        conn.commit()

        return jsonify({"message": "ok"}), 200
    else:
        return jsonify({"message": "user not found"}), 404


# 定义路由
@users_api.route("/photo", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录（桌面）
    file_path = r'templates/image'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        avatar = f'image/{file_name}'
        return avatar
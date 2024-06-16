from datetime import datetime

import pymysql.cursors
from flask import Blueprint, request, jsonify
from app.database import get_mysql_connection

auth_api = Blueprint('auth_api', __name__)

@auth_api.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    confirm_password = data.get('confirmPassword')
    email = data.get('email')

    if not username or not password or not confirm_password or not email:
        return jsonify({"message": "所有字段都是必填项"}), 201

    if password != confirm_password:
        return jsonify({"message": "两次输入的密码不一致"}), 201

    conn = get_mysql_connection()
    cursor = conn.cursor()

    # 检查用户名是否已存在
    cursor.execute("SELECT * FROM User WHERE username=%s", (username,))
    existing_user = cursor.fetchone()
    if existing_user:
        return jsonify({"message": "用户名已存在"}), 201

    # 检查邮箱是否已存在
    cursor.execute("SELECT * FROM user WHERE email=%s", (email,))
    existing_email = cursor.fetchone()
    if existing_email:
        return jsonify({"message": "邮箱已存在"}), 201

    # 将用户信息插入数据库
    registration_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO User (username, password, email, registration_time) VALUES (%s, %s, %s, %s)",
                   (username, password, email, registration_time))
    conn.commit()
    conn.close()

    return jsonify({"message": "register ok"}), 200

@auth_api.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "所有字段都是必填项"}), 201

    conn = get_mysql_connection()
    cursor = conn.cursor()

    # 查询用户是否存在并验证密码
    cursor.execute("SELECT * FROM User WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()
    if user:
        conn.close()
        return jsonify({"message": "login ok"}), 200

    return jsonify({"message": "登录失败，请检查您的凭据。"}), 201

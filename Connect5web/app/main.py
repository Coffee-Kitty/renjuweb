from flask import Flask, send_file, redirect
from flask_sockets import Sockets

from app.front import front_api
from app.game.history import history_api
from app.game.pk import pk_api
from app.game.renju import renju_api
from app.users.users import users_api
from flask_cors import CORS
app = Flask(__name__)


CORS(app)  # 将 CORS 应用到 Flask 应用上
# # 数据库 和 加密
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# 添加路由信息
from app.auth.auth import auth_api

app.register_blueprint(front_api)
# app.register_blueprint(auth_api)
app.register_blueprint(auth_api, url_prefix='/auths')
app.register_blueprint(users_api, url_prefix='/users')
app.register_blueprint(renju_api, url_prefix='/renju')
app.register_blueprint(pk_api, url_prefix='/pk')
app.register_blueprint(history_api, url_prefix='/history')

if __name__ == "__main__":
    app.run(debug=True, port=5500)


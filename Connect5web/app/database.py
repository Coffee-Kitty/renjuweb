
import pymysql
# 配置MySQL连接
def get_mysql_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='123456',
        db='connect5web',
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True  # 启用自动提交
    )
#  ✅ db/board_util.py
import pymysql

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='humanda5',
        password='humanda5',
        database='demoweb', 
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def insert_post(title, content):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO board (title, content) VALUES (%s, %s)"
            cursor.execute(sql, (title, content))
        conn.commit()
    finally:
        conn.close()
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


def get_posts_by_page(page, per_page=10):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            offset = (page - 1) * per_page
            cursor.execute("SELECT COUNT(*) as total FROM board")
            total = cursor.fetchone()['total']

            cursor.execute("""
                SELECT id, title, created_at
                FROM board
                ORDER BY id DESC
                LIMIT %s OFFSET %s
            """, (per_page, offset))
            posts = cursor.fetchall()
        return posts, total
    finally:
        conn.close()


def insert_post(title, content):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO board (title, content) VALUES (%s, %s)"
            cursor.execute(sql, (title, content))
        conn.commit()
    finally:
        conn.close()


def get_post_by_id(post_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM board WHERE id = %s"
            cursor.execute(sql, (post_id,))
            return cursor.fetchone()
    finally:
        conn.close()


def update_post(post_id, title, content):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE board SET title = %s, content = %s WHERE id = %s"
            cursor.execute(sql, (title, content, post_id))
        conn.commit()
    finally:
        conn.close()


def delete_post_by_id(post_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM board WHERE id = %s"
            cursor.execute(sql, (post_id,))
        conn.commit()
    finally:
        conn.close()

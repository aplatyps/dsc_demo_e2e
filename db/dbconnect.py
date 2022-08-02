import sqlite3

def create_connection(db_file=None):
    if db_file == None:
        db_file = 'demo_database.db'
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        conn = None
        print(e)
    return conn


def close_connection(conn):
    if conn:
        conn.commit()
        conn.close()
    return
import sqlite3
import datetime

DB_NAME = "chat_history.db"

def init_db():
    """Initializes the SQLite database and creates the chat_logs table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pdf_name TEXT,
            question TEXT,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_chat(question, answer, pdf_name="Unknown"):
    """Logs a single chat interaction to the database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''
            INSERT INTO chat_logs (timestamp, pdf_name, question, answer)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, pdf_name, question, answer))
        conn.commit()
        conn.close()
    except:
        pass # Silent fail to ensure chat continues even if DB errors occur

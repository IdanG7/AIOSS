import sqlite3


def create_password_db(password_file, db_name="data/passwords.db"):
    """
    Create an SQLite database with the world's top passwords.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS common_passwords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            password TEXT UNIQUE
        )
    """
    )

    # Insert passwords
    with open(password_file, "r", encoding="utf-8") as file:
        for line in file:
            password = line.strip()
            cursor.execute(
                "INSERT OR IGNORE INTO common_passwords (password) VALUES (?)",
                (password,),
            )

    conn.commit()
    conn.close()


def check_common_passwords(db_name, password):
    """
    Check if the password exists in the SQLite database.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM common_passwords WHERE password = ?", (password,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

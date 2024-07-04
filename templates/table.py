import sqlite3

# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('users.db')

# Create a cursor object
c = conn.cursor()

# Create the users table
c.execute('''CREATE TABLE users (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             fullname TEXT NOT NULL,
             email TEXT NOT NULL UNIQUE,
             username TEXT NOT NULL UNIQUE,
             password TEXT NOT NULL)''')

# Commit the changes and close the connection
conn.commit()
conn.close()

import sqlite3

conn = sqlite3.connect("bullwhip.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS weather(
id INTEGER PRIMARY KEY AUTOINCREMENT,
date TEXT,
city TEXT,
temperature REAL,
wind REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS news(
id INTEGER PRIMARY KEY AUTOINCREMENT,
title TEXT,
date TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS commodity(
id INTEGER PRIMARY KEY AUTOINCREMENT,
product TEXT,
price REAL,
unit TEXT,
date TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS risk(
id INTEGER PRIMARY KEY AUTOINCREMENT,
supply REAL,
demand REAL,
inventory REAL,
overall REAL,
date TEXT
)
""")

conn.commit()

conn.close()

print("Database Created Successfully")

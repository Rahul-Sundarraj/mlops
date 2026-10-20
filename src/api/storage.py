"""SQLite storage for predictions, GPS pings and complaints."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY, train_no INTEGER, station TEXT, date TEXT, predicted_delay REAL, created_at TEXT);
CREATE TABLE IF NOT EXISTS pings (
    id INTEGER PRIMARY KEY, train_no INTEGER, station TEXT, date TEXT, actual_delay REAL, created_at TEXT);
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY, train_no INTEGER, date TEXT, text TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_pred ON predictions(train_no, station, date);
CREATE INDEX IF NOT EXISTS ix_ping ON pings(train_no, date);
CREATE INDEX IF NOT EXISTS ix_comp ON complaints(train_no, date);
"""


def connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn

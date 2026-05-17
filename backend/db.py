"""
db.py
-----
SQLite-backed plate database and session history store.

Tables:
  plates   — known vehicle records with status (clear / flagged / expired)
  sessions — history of every OCR analysis run in this session
"""

import sqlite3
import os
import json
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(__file__), 'ocr_data.db')


def _get_conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and seed some demo plate records."""
    conn = _get_conn()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            owner_name  TEXT,
            vehicle_type TEXT,
            status      TEXT DEFAULT 'clear',   -- clear | flagged | expired
            note        TEXT,
            registered  TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            filename     TEXT,
            plate_text   TEXT,
            plate_matched INTEGER,
            avg_confidence REAL,
            blur_type    TEXT,
            sharpness    REAL,
            retry_count  INTEGER
        )
    ''')

    # Seed demo records (only if table is empty)
    c.execute('SELECT COUNT(*) FROM plates')
    if c.fetchone()[0] == 0:
        seed = [
            ('MH12AB1234', 'Rahul Sharma',    'Car',      'clear',   None,                        '2020-03-15'),
            ('DL3CAB1234', 'Priya Patel',     'Car',      'clear',   None,                        '2019-07-22'),
            ('KA05MJ7890', 'Suresh Kumar',    'Bike',     'flagged', 'Reported stolen 2024-01-10', '2021-11-05'),
            ('TN22BZ4567', 'Anita Nair',      'Truck',    'expired', 'Registration expired',      '2017-04-30'),
            ('MH01AP9999', 'Vijay Transport', 'Truck',    'flagged', 'Multiple traffic violations','2022-08-19'),
            ('GJ05AC3344', 'Deepak Shah',     'Car',      'clear',   None,                        '2023-02-11'),
            ('UP32BT6600', 'Mohan Yadav',     'Car',      'flagged', 'Outstanding challans',      '2018-09-03'),
            ('HR26DA8080', 'Ravi Verma',      'SUV',      'clear',   None,                        '2021-06-25'),
        ]
        c.executemany(
            'INSERT INTO plates (plate_number, owner_name, vehicle_type, status, note, registered) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            seed
        )

    conn.commit()
    conn.close()


def lookup_plate(plate_text):
    """
    Query the plates table for a given plate number.
    Returns a dict with plate info, or None if not found.
    """
    if not plate_text:
        return None
    cleaned = plate_text.strip().upper().replace(' ', '').replace('-', '')
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM plates WHERE REPLACE(REPLACE(plate_number," ",""),"-","") = ?', (cleaned,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def save_session(filename, plate_text, plate_matched, avg_confidence,
                 blur_type, sharpness, retry_count):
    """Append a new row to the sessions history table."""
    conn = _get_conn()
    conn.execute(
        '''INSERT INTO sessions
           (timestamp, filename, plate_text, plate_matched, avg_confidence, blur_type, sharpness, retry_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            datetime.utcnow().isoformat(),
            filename or '',
            plate_text or '',
            1 if plate_matched else 0,
            avg_confidence or 0.0,
            blur_type or '',
            sharpness or 0.0,
            retry_count or 0,
        )
    )
    conn.commit()
    conn.close()


def get_history(limit=50):
    """Return the last `limit` session records, most recent first."""
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM sessions ORDER BY id DESC LIMIT ?', (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

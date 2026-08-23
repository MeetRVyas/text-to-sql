"""
sample_db/create_sample_db.py
------------------------------
Creates a small bundled SQLite database (concert_singer) for local
development and API smoke-testing without the full Spider download.

Usage:
    python sample_db/create_sample_db.py
"""

import sqlite3
from pathlib import Path


SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS stadium (
    stadium_id   INTEGER PRIMARY KEY,
    location     TEXT,
    name         TEXT,
    capacity     INTEGER,
    highest      INTEGER,
    lowest       INTEGER,
    average      INTEGER
);

CREATE TABLE IF NOT EXISTS singer (
    singer_id    INTEGER PRIMARY KEY,
    name         TEXT,
    country      TEXT,
    song_name    TEXT,
    song_release_year TEXT,
    age          INTEGER,
    is_male      TEXT
);

CREATE TABLE IF NOT EXISTS concert (
    concert_id   INTEGER PRIMARY KEY,
    concert_name TEXT,
    theme        TEXT,
    stadium_id   INTEGER,
    year         TEXT,
    FOREIGN KEY (stadium_id) REFERENCES stadium(stadium_id)
);

CREATE TABLE IF NOT EXISTS singer_in_concert (
    concert_id   INTEGER,
    singer_id    INTEGER,
    PRIMARY KEY (concert_id, singer_id),
    FOREIGN KEY (concert_id) REFERENCES concert(concert_id),
    FOREIGN KEY (singer_id) REFERENCES singer(singer_id)
);
"""

SEED_DATA = """
INSERT INTO stadium VALUES (1, 'Raith Rovers', 'Stark''s Park', 10104, 4812, 1294, 2106);
INSERT INTO stadium VALUES (2, 'Ayr United',   'Somerset Park', 11998, 2363, 1057, 1477);
INSERT INTO stadium VALUES (3, 'East Fife',    'Bayview Stadium', 2000, 1980, 533, 864);

INSERT INTO singer VALUES (1, 'Joe Sharp',    'Netherlands',    'You',              '1992', 52, 'F');
INSERT INTO singer VALUES (2, 'Timbaland',    'United States',  'Apologize',        '2006', 32, 'M');
INSERT INTO singer VALUES (3, 'Justin Brown',  'France',        'Love',             '2000', 29, 'M');

INSERT INTO concert VALUES (1, 'Glastonbury', 'Pop', 1, '2014');
INSERT INTO concert VALUES (2, 'T in the Park', 'Rock', 2, '2015');
INSERT INTO concert VALUES (3, 'Lollapalooza', 'Indie', 3, '2016');

INSERT INTO singer_in_concert VALUES (1, 1);
INSERT INTO singer_in_concert VALUES (1, 2);
INSERT INTO singer_in_concert VALUES (2, 3);
INSERT INTO singer_in_concert VALUES (3, 1);
"""


def create_sample_db(output_dir: str = "data/spider/database/concert_singer") -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "concert_singer.db"

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SQL_SCHEMA)
    conn.executescript(SEED_DATA)
    conn.commit()
    conn.close()

    print(f"Sample DB created: {db_path}")
    return str(db_path)


if __name__ == "__main__":
    create_sample_db()
